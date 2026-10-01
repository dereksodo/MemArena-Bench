"""DailyScheduler: merge encounter windows + events into a conversation schedule.

Phase 3 of the lifecycle simulation replaces the pure event-driven simulation
loop with a scheduler-driven approach.  Encounter windows (detected from
skeleton schedules) become the *primary* conversation trigger; events provide
narrative context.

Four phases:
  A. Group meetings — guaranteed whenever a group's meeting_schedule matches
     an encounter window where all required members are co-located.
  B. Event-triggered encounters — encounter windows that overlap a world event
     inherit that event as conversational context.
  C. Casual encounters — remaining encounter windows are probabilistically
     converted to conversations based on location type, familiarity, and
     shared group membership.
  D. Remote conversations — phone calls / text chats between graph-connected
     agents during overlapping solo time.  No co-location needed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from MASim.core.agent import EntityAgent
from MASim.core.schema import (
    ActivityEntry, EncounterWindow, Group, Location, WorldEvent,
)
from MASim.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# ConversationSpec — one scheduled conversation
# ---------------------------------------------------------------------------

@dataclass
class ConversationSpec:
    """A single conversation to execute in the scheduler-driven loop."""
    spec_id: str = ""
    participants: List[str] = field(default_factory=list)   # agent_ids
    location_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    trigger: str = "encounter"      # "group_meeting" | "encounter" | "event" | "routine_facet"
    group_id: str = ""
    event: Optional[WorldEvent] = None  # narrative context if available
    modality: str = "face_to_face"
    is_group: bool = False
    interest_domain: str = ""       # interest domain for faceted sessions
    session_type: str = ""          # "routine" | "trigger" | "" (legacy)
    routine_prompt: str = ""        # pre-generated conversation prompt (routine sessions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "participants": list(self.participants),
            "location_id": self.location_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "trigger": self.trigger,
            "group_id": self.group_id,
            "event_id": self.event.event_id if self.event else "",
            "modality": self.modality,
            "is_group": self.is_group,
            "interest_domain": self.interest_domain,
            "session_type": self.session_type,
            "routine_prompt": self.routine_prompt,
        }


# ---------------------------------------------------------------------------
# Encounter probability modifiers (§5.3 of design doc)
# ---------------------------------------------------------------------------

_LOCATION_TYPE_MODIFIER: Dict[str, float] = {
    "third_place_regular": 0.8,     # café, bar — highly social
    "third_place_event": 0.7,
    "workplace": 0.6,
    "outdoor": 0.5,
    "home_shared": 0.5,
    "service": 0.3,
    "transit": 0.2,
    "home_private": 0.1,            # rarely triggers casual chat with non-residents
}

_DUNBAR_FAMILIARITY_BONUS: Dict[int, float] = {
    1: 2.0,     # intimate circle (5 people)
    2: 1.5,     # close friends (15)
    3: 1.2,     # active network (50)
    4: 1.0,     # acquaintances (150)
}
_STRANGER_FAMILIARITY = 0.4

_SHARED_GROUP_BONUS = 0.3


# ---------------------------------------------------------------------------
# Role-context modifiers (DESIGN §4.2)
# ---------------------------------------------------------------------------

ROLE_CONTEXT_MODIFIERS: Dict[str, Dict[str, Any]] = {
    "family": {
        "register": "intimate",
        "topic_bias": ["home life", "family news", "health", "emotional support"],
        "vocabulary_note": "Uses nicknames, doesn't perform. More emotionally direct.",
    },
    "work_team": {
        "register": "professional",
        "topic_bias": ["project status", "decisions", "technical problems", "deadlines"],
        "vocabulary_note": "Uses field jargon. Filters personal life out unless close colleagues.",
    },
    "hobby": {
        "register": "casual-enthusiastic",
        "topic_bias": ["shared hobby", "events", "gear or technique", "plans to meet up"],
        "vocabulary_note": "Playful. May use in-group slang. Speaks with passion.",
    },
    "neighbourhood": {
        "register": "civil-casual",
        "topic_bias": ["local issues", "weather", "property", "community events"],
        "vocabulary_note": "Polite but direct. Careful with conflict.",
    },
    "civic": {
        "register": "formal-civic",
        "topic_bias": ["community decisions", "volunteering", "planning", "public issues"],
        "vocabulary_note": "Measured tone. Tends toward consensus-building language.",
    },
    "friendship_circle": {
        "register": "casual-intimate",
        "topic_bias": ["gossip", "plans", "personal updates", "shared memories"],
        "vocabulary_note": "Relaxed and open. Inside jokes welcome.",
    },
}


def build_role_context_prefix(
    *,
    group: Optional["Group"],
    role_name: str,
    location: Optional["Location"],
) -> str:
    """Build a multi-line context prefix describing location, group, and role.

    Returns ``""`` when no meaningful context is available.
    Callers can unconditionally prepend the result.
    """
    parts: List[str] = []

    # Location context
    if location and location.name:
        loc_label = location.location_type.replace("_", " ") if location.location_type else "location"
        parts.append(f"You're at {location.name} ({loc_label}).")

    # Group + role context
    if group and group.name:
        group_type = group.group_type or "group"
        role_clause = f" — you're here in your role as {role_name}" if role_name else ""
        parts.append(
            f"This is a {group.name} ({group_type}) context{role_clause}."
        )

        modifiers = ROLE_CONTEXT_MODIFIERS.get(group_type, {})
        if modifiers.get("vocabulary_note"):
            parts.append(f"Register: {modifiers['vocabulary_note']}")
        topics = modifiers.get("topic_bias")
        if topics:
            parts.append(f"Topics likely to come up: {', '.join(topics)}.")

    if not parts:
        return ""
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# DailyScheduler
# ---------------------------------------------------------------------------

class DailyScheduler:
    """Merge encounter windows and events into a unified conversation schedule."""

    def __init__(
        self,
        groups: List[Group],
        agents: Dict[str, EntityAgent],
        locations: List[Location],
        rng: np.random.Generator,
        modality_weights: Dict[str, float],
        base_encounter_prob: float = 0.15,
        graph: Optional[Any] = None,
    ):
        self.groups = groups
        self.agents = agents
        self.rng = rng
        self.modality_weights = modality_weights
        self.base_encounter_prob = base_encounter_prob
        self.graph = graph  # social graph for minimum-session guarantee

        # Pre-compute lookup tables
        self._group_by_id: Dict[str, Group] = {g.group_id: g for g in groups}
        self._location_map: Dict[str, Location] = {l.location_id: l for l in locations}
        self._agent_groups: Dict[str, Set[str]] = {}  # agent_id -> set of group_ids
        for g in groups:
            for member_id in g.members:
                self._agent_groups.setdefault(member_id, set()).add(g.group_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_day_schedule(
        self,
        encounter_windows: List[EncounterWindow],
        events: List[WorldEvent],
        day_time_range: Tuple[float, float],
        day_length: float,
        targets: Dict[str, int],
        *,
        skeleton_logs: Optional[Dict[str, List[ActivityEntry]]] = None,
    ) -> List[ConversationSpec]:
        """Generate PP specs for a single day using proportion-based targets.

        *targets* maps session type to count:
        ``{"event_triggered": N, "encounter": N, "remote": N}``
        (facet_routine is handled separately by FacetScheduler).

        Group meetings are always included (uncapped by proportion).
        If any type under-produces, encounter fills the gap.
        """
        specs: List[ConversationSpec] = []
        consumed_window_ids: Set[str] = set()

        # Phase A: Group meetings (always included, not counted in proportions)
        group_specs, consumed_a = self._phase_a_group_meetings(
            encounter_windows, time_range=day_time_range, day_length=day_length,
            target_count=0,  # no cap for group meetings
        )
        specs.extend(group_specs)
        consumed_window_ids.update(consumed_a)

        # Phase B: Event-triggered encounters
        # Prefer same-day events; if too few, sample from all past/current events
        # (people discuss recent news even days later).
        day_events = [
            e for e in events
            if day_time_range[0] <= e.timestamp < day_time_range[1]
        ]
        event_target = targets.get("event_triggered", 0)
        if len(day_events) < event_target and events:
            # Supplement with events from other days (most recent first)
            past_events = [
                e for e in events
                if e.timestamp < day_time_range[1] and e not in day_events
            ]
            past_events.sort(key=lambda e: e.timestamp, reverse=True)
            day_events.extend(past_events[:event_target - len(day_events)])
        event_specs, consumed_b = self._phase_b_event_encounters(
            encounter_windows, day_events,
            target_count=event_target,
            day_length=day_length,
        )
        specs.extend(event_specs)
        consumed_window_ids.update(consumed_b)

        # Phase C: Casual encounters (from encounter windows in this day)
        day_windows = [
            w for w in encounter_windows
            if w.window_id not in consumed_window_ids
            and w.start_time >= day_time_range[0]
            and w.start_time < day_time_range[1]
        ]
        encounter_target = targets.get("encounter", 0)
        casual_specs = self._phase_c_casual_encounters(day_windows)
        # Supplement with graph-based f2f if casual encounters are insufficient
        if len(casual_specs) < encounter_target and self.graph is not None:
            extra_f2f = self._phase_c2_graph_encounters(
                existing_specs=specs + casual_specs,
                target_count=encounter_target - len(casual_specs),
                time_range=day_time_range,
                max_sessions_per_day=0,
                day_length=day_length,
            )
            casual_specs.extend(extra_f2f)
        if len(casual_specs) > encounter_target:
            cap_rng = np.random.default_rng(self.rng.integers(2**63))
            indices = cap_rng.choice(len(casual_specs), encounter_target, replace=False)
            casual_specs = [casual_specs[i] for i in sorted(indices)]
        specs.extend(casual_specs)

        # Phase D: Remote conversations
        remote_target = targets.get("remote", 0)
        remote_specs: List[ConversationSpec] = []
        if skeleton_logs and self.graph is not None:
            remote_specs = self._phase_d_remote_conversations(
                skeleton_logs=skeleton_logs,
                existing_specs=specs,
                target_count=remote_target,
                time_range=day_time_range,
                max_sessions_per_day=0,
                day_length=day_length,
            )
        specs.extend(remote_specs)

        specs.sort(key=lambda s: s.start_time)
        return specs

    def build_conversation_schedule(
        self,
        encounter_windows: List[EncounterWindow],
        events: List[WorldEvent],
        max_sessions: int,
        *,
        skeleton_logs: Optional[Dict[str, List[ActivityEntry]]] = None,
        time_range: Optional[Tuple[float, float]] = None,
        max_sessions_per_day: int = 0,
        day_length: float = 1.0,
    ) -> List[ConversationSpec]:
        """Merge encounter windows and events into a conversation schedule.

        Five phases with target ratio group_meeting:event:encounter:graph_f2f:remote
        = 1:10:10:5:20.  Each phase is capped at its ratio-derived target; the
        overall total is capped at max_sessions.
        """
        # Compute per-type targets from fixed ratio
        _RATIO = {
            "group_meeting": 1, "event": 10, "encounter": 10,
            "graph_f2f": 5, "remote": 20,
        }
        total_ratio = sum(_RATIO.values())
        targets = {k: max(1, int(max_sessions * v / total_ratio))
                   for k, v in _RATIO.items()}
        log.info("DailyScheduler targets: %s (max_sessions=%d)", targets, max_sessions)

        specs: List[ConversationSpec] = []
        consumed_window_ids: Set[str] = set()

        # Phase A: Group meetings
        group_specs, consumed_a = self._phase_a_group_meetings(
            encounter_windows, time_range=time_range, day_length=day_length,
            target_count=targets["group_meeting"],
        )
        specs.extend(group_specs)
        consumed_window_ids.update(consumed_a)

        # Phase B: Event-triggered encounters (generated directly from events)
        event_specs, consumed_b = self._phase_b_event_encounters(
            encounter_windows, events,
            target_count=targets["event"],
            day_length=day_length,
        )
        specs.extend(event_specs)
        consumed_window_ids.update(consumed_b)

        # Phase C: Casual encounters (from encounter windows)
        remaining_windows = [
            w for w in encounter_windows if w.window_id not in consumed_window_ids
        ]
        casual_specs = self._phase_c_casual_encounters(remaining_windows)
        if len(casual_specs) > targets["encounter"]:
            cap_rng = np.random.default_rng(self.rng.integers(2**63))
            indices = cap_rng.choice(len(casual_specs), targets["encounter"], replace=False)
            casual_specs = [casual_specs[i] for i in sorted(indices)]
        specs.extend(casual_specs)

        # Phase C2: Graph-based face-to-face encounters
        graph_f2f_specs: List[ConversationSpec] = []
        if self.graph is not None and time_range:
            graph_f2f_specs = self._phase_c2_graph_encounters(
                existing_specs=specs,
                target_count=targets["graph_f2f"],
                time_range=time_range,
                max_sessions_per_day=max_sessions_per_day,
                day_length=day_length,
            )
            specs.extend(graph_f2f_specs)

        # Phase D: Remote conversations (fill solo time with phone/text)
        remote_specs: List[ConversationSpec] = []
        if skeleton_logs and time_range and self.graph is not None:
            remote_specs = self._phase_d_remote_conversations(
                skeleton_logs=skeleton_logs,
                existing_specs=specs,
                target_count=targets["remote"],
                time_range=time_range,
                max_sessions_per_day=max_sessions_per_day,
                day_length=day_length,
            )
            specs.extend(remote_specs)

        # Sort by start_time, cap at max_sessions
        specs.sort(key=lambda s: s.start_time)
        pre_cap = len(specs)
        if len(specs) > max_sessions:
            specs = specs[:max_sessions]

        log.info(
            "DailyScheduler: %d group_meeting, %d event, %d encounter, "
            "%d graph_f2f, %d remote → %d specs (pre-cap %d, cap %d)",
            len(group_specs), len(event_specs), len(casual_specs),
            len(graph_f2f_specs), len(remote_specs),
            len(specs), pre_cap, max_sessions,
        )
        return specs

    # ------------------------------------------------------------------
    # Phase A: Group meetings
    # ------------------------------------------------------------------

    def _phase_a_group_meetings(
        self,
        encounter_windows: List[EncounterWindow],
        time_range: Optional[Tuple[float, float]] = None,
        day_length: float = 1.0,
        target_count: int = 0,
    ) -> Tuple[List[ConversationSpec], Set[str]]:
        """Generate group meeting specs from group schedules.

        Each group with a meeting_schedule gets one spec per scheduled
        meeting mapped onto the simulation time range.  No longer depends
        on encounter windows (groups schedule their own meetings).

        Any encounter windows that happen to overlap a generated meeting
        are marked as consumed so they aren't double-counted.
        """
        specs: List[ConversationSpec] = []
        consumed: Set[str] = set()

        # Determine simulation time range
        if time_range:
            t_start, t_end = time_range
        elif encounter_windows:
            t_start = min(w.start_time for w in encounter_windows)
            t_end = max(w.end_time for w in encounter_windows)
        else:
            t_start, t_end = 0.0, 1000.0
        t_span = max(t_end - t_start, 1.0)
        # day_length is now passed from config rather than auto-computed

        for group in self.groups:
            if not group.meeting_schedule:
                continue
            member_ids = set(group.members.keys())
            # Only keep members that are actual agents
            member_ids = {m for m in member_ids if m in self.agents}
            if len(member_ids) < 2:
                continue

            meeting_loc = group.home_location_id

            for sched in group.meeting_schedule:
                freq = sched.get("frequency", "weekly")
                start_hour = sched.get("start_hour", 10.0)
                duration = sched.get("duration_hours", 1.0)

                # Map meetings onto simulation days
                if freq == "weekday":
                    # ~5 per 7 days → every day_length for simplicity
                    interval = day_length
                elif freq == "weekly":
                    interval = day_length * 7
                elif freq == "monthly":
                    interval = day_length * 30
                else:
                    interval = day_length * 7

                # Generate one meeting per interval
                # Place the first meeting at hour offset within the first interval
                hour_frac = start_hour / 24.0
                meeting_time = t_start + hour_frac * day_length
                while meeting_time < t_end:
                    m_end = meeting_time + (duration / 24.0) * day_length
                    specs.append(ConversationSpec(
                        spec_id=f"spec_grp_{uuid.uuid4().hex[:8]}",
                        participants=sorted(member_ids),
                        location_id=meeting_loc,
                        start_time=meeting_time,
                        end_time=min(m_end, t_end),
                        trigger="group_meeting",
                        group_id=group.group_id,
                        modality="face_to_face",
                        is_group=True,
                    ))
                    meeting_time += interval

        # Cap at ratio-derived target
        if target_count > 0 and len(specs) > target_count:
            pre_cap = len(specs)
            cap_rng = np.random.default_rng(self.rng.integers(2**63))
            indices = cap_rng.choice(len(specs), target_count, replace=False)
            specs = [specs[i] for i in sorted(indices)]
            log.info("Phase A: capped group meetings from %d to %d", pre_cap, target_count)

        return specs, consumed

    # ------------------------------------------------------------------
    # Phase B: Event-triggered encounters
    # ------------------------------------------------------------------

    def _phase_b_event_encounters(
        self,
        encounter_windows: List[EncounterWindow],
        events: List[WorldEvent],
        target_count: int = 0,
        day_length: float = 1.0,
    ) -> Tuple[List[ConversationSpec], Set[str]]:
        """Generate conversation specs directly from events.

        Each event produces one or more pairwise conversations among its
        visible agents.  For dyadic/private events with <=2 visible agents,
        one spec is created.  For community/global events, multiple pairs
        are sampled from the visibility mask so that the event ripples
        through the social network.
        """
        from itertools import combinations

        specs: List[ConversationSpec] = []
        consumed: Set[str] = set()

        for evt in events:
            # Filter to agents that actually exist in the simulation
            visible = sorted(a for a in evt.visibility_mask if a in self.agents)
            if len(visible) < 2:
                continue

            location = evt.location_id or ""
            conv_duration = day_length * 0.04  # ~1 hour in a 24h day

            evt_domain = getattr(evt, "interest_domain", "") or ""

            if evt.category.value in ("dyadic", "private"):
                # One conversation among the visible agents (usually 2-3)
                specs.append(ConversationSpec(
                    spec_id=f"spec_evt_{uuid.uuid4().hex[:8]}",
                    participants=visible,
                    location_id=location,
                    start_time=evt.timestamp,
                    end_time=evt.timestamp + conv_duration,
                    trigger="event",
                    event=evt,
                    modality="face_to_face",
                    is_group=len(visible) > 2,
                    interest_domain=evt_domain,
                    session_type="trigger",
                ))
            else:
                # community / global: sample multiple pairs so the event
                # generates several conversations (people discuss the news)
                all_pairs = list(combinations(visible, 2))
                self.rng.shuffle(all_pairs)
                # Use up to 5 pairs per event, spread across a short window
                n_pairs = min(len(all_pairs), 5)
                for i, (a, b) in enumerate(all_pairs[:n_pairs]):
                    t_offset = i * conv_duration * 0.5
                    specs.append(ConversationSpec(
                        spec_id=f"spec_evt_{uuid.uuid4().hex[:8]}",
                        participants=sorted([a, b]),
                        location_id=location,
                        start_time=evt.timestamp + t_offset,
                        end_time=evt.timestamp + t_offset + conv_duration,
                        trigger="event",
                        event=evt,
                        modality=self._roll_modality(),
                        is_group=False,
                        interest_domain=evt_domain,
                        session_type="trigger",
                    ))

        # Cap at target
        if target_count > 0 and len(specs) > target_count:
            cap_rng = np.random.default_rng(self.rng.integers(2**63))
            indices = cap_rng.choice(len(specs), target_count, replace=False)
            specs = [specs[i] for i in sorted(indices)]

        return specs, consumed

    # ------------------------------------------------------------------
    # Phase C: Casual encounters
    # ------------------------------------------------------------------

    def _phase_c_casual_encounters(
        self,
        encounter_windows: List[EncounterWindow],
    ) -> List[ConversationSpec]:
        """Convert remaining encounter windows to casual conversations
        based on probabilistic filtering.

        For multi-agent windows, evaluate ALL possible pairs so that
        co-location opportunities produce multiple face-to-face sessions.
        """
        from itertools import combinations

        specs: List[ConversationSpec] = []

        for window in encounter_windows:
            agents_in_window = window.agent_ids
            if len(agents_in_window) < 2:
                continue

            # Evaluate all pairs in this window
            pairs = list(combinations(agents_in_window, 2))
            self.rng.shuffle(pairs)

            for a, b in pairs:
                p = self._encounter_probability(a, b, window.location_id)
                if self.rng.random() >= p:
                    continue

                modality = self._roll_modality()
                specs.append(ConversationSpec(
                    spec_id=f"spec_enc_{uuid.uuid4().hex[:8]}",
                    participants=sorted([a, b]),
                    location_id=window.location_id,
                    start_time=window.start_time,
                    end_time=window.end_time,
                    trigger="encounter",
                    modality=modality,
                    is_group=False,
                ))

        return specs

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Phase C2: Graph-based face-to-face encounters
    # ------------------------------------------------------------------

    def _phase_c2_graph_encounters(
        self,
        existing_specs: List[ConversationSpec],
        target_count: int,
        time_range: Tuple[float, float],
        max_sessions_per_day: int,
        day_length: float = 1.0,
    ) -> List[ConversationSpec]:
        """Generate face-to-face encounter specs for graph-connected agents.

        Spreads conversations across the simulation time range to ensure
        sufficient face-to-face sessions even when encounter windows are
        scarce (e.g. agents have unique locations in dry-run mode).

        Budget is set by the ratio-derived target_count.
        """
        if self.graph is None:
            return []

        t_start, t_end = time_range
        t_span = max(t_end - t_start, 1.0)

        n_needed = target_count
        if n_needed <= 0:
            return []

        # Build scored candidates from graph edges
        candidates: List[Tuple[float, str, str]] = []
        for u, v, data in self.graph.edges(data=True):
            if u not in self.agents or v not in self.agents:
                continue
            edge_weight = data.get("weight", 0.5)
            shared_group = 1.3 if self._share_group(u, v) else 1.0
            score = edge_weight * shared_group + float(self.rng.uniform(0, 0.01))
            candidates.append((score, u, v))

        candidates.sort(key=lambda c: c[0], reverse=True)

        # Track per-day counts
        day_counts: Dict[int, int] = {}
        for spec in existing_specs:
            d = int((spec.start_time - t_start) / day_length)
            day_counts[d] = day_counts.get(d, 0) + 1

        n_days = max(1, int(np.ceil(t_span / day_length)))
        specs: List[ConversationSpec] = []

        # Distribute needed specs across days, cycling through candidates
        cand_idx = 0
        for day_idx in range(n_days):
            if len(specs) >= n_needed:
                break
            # How many to add this day
            day_target = max(1, n_needed // n_days)
            day_total = day_counts.get(day_idx, 0)

            for _ in range(day_target):
                if len(specs) >= n_needed:
                    break
                if max_sessions_per_day > 0 and day_total >= max_sessions_per_day:
                    break
                if not candidates:
                    break

                # Pick next candidate (wrap around)
                score, u, v = candidates[cand_idx % len(candidates)]
                cand_idx += 1

                # Place at a random time within this day
                day_start = t_start + day_idx * day_length
                day_end = min(day_start + day_length, t_end)
                t = day_start + float(self.rng.uniform(0.1, 0.9)) * (day_end - day_start)

                specs.append(ConversationSpec(
                    spec_id=f"spec_gf2f_{uuid.uuid4().hex[:8]}",
                    participants=sorted([u, v]),
                    location_id="",  # no specific location
                    start_time=t,
                    end_time=t + 1.0,
                    trigger="encounter",
                    modality="face_to_face",
                    is_group=False,
                ))
                day_total += 1
                day_counts[day_idx] = day_total

        log.info(
            "Phase C2: generated %d graph-based f2f specs (target=%d)",
            len(specs), n_needed,
        )
        return specs

    # ------------------------------------------------------------------
    # Phase D: Remote conversations (phone / text during solo time)
    # ------------------------------------------------------------------

    def _phase_d_remote_conversations(
        self,
        skeleton_logs: Dict[str, List[ActivityEntry]],
        existing_specs: List[ConversationSpec],
        target_count: int,
        time_range: Tuple[float, float],
        max_sessions_per_day: int,
        day_length: float = 1.0,
    ) -> List[ConversationSpec]:
        """Generate remote PP conversations between graph-connected agents
        during overlapping solo time intervals.

        Budget is set by the ratio-derived target_count.
        """
        if self.graph is None:
            return []

        t_start, t_end = time_range
        t_span = max(t_end - t_start, 1.0)
        n_days = int(np.ceil(t_span / day_length))
        min_overlap = 0.05 * day_length  # 5% of a day

        # 1. Build solo-interval index per agent per day
        #    {agent_id: {day_idx: [(start, end), ...]}}
        solo_index: Dict[str, Dict[int, List[Tuple[float, float]]]] = {}
        for agent_id, entries in skeleton_logs.items():
            day_map: Dict[int, List[Tuple[float, float]]] = {}
            for e in entries:
                if e.activity_type != "solo":
                    continue
                day_idx = int((e.start_time - t_start) / day_length)
                day_map.setdefault(day_idx, []).append((e.start_time, e.end_time))
            if day_map:
                solo_index[agent_id] = day_map

        # 2. Track existing per-day counts and per-agent load
        day_counts: Dict[int, int] = {}
        agent_load: Dict[str, int] = {}
        for spec in existing_specs:
            d = int((spec.start_time - t_start) / day_length)
            day_counts[d] = day_counts.get(d, 0) + 1
            for pid in spec.participants:
                agent_load[pid] = agent_load.get(pid, 0) + 1

        max_remote = target_count
        log.info(
            "Phase D: target=%d, agents_with_solo=%d/%d, n_days=%d, day_length=%.2f",
            max_remote, len(solo_index), len(self.agents), n_days, day_length,
        )
        if max_remote <= 0:
            return []

        # 3. Build scored candidates for every (edge, day) pair
        candidates: List[Tuple[float, str, str, int, float, float]] = []
        # (score, u, v, day_idx, overlap_start, overlap_end)

        for u, v, data in self.graph.edges(data=True):
            if u not in solo_index or v not in solo_index:
                continue
            edge_weight = data.get("weight", 0.5)
            shared_group = 1.3 if self._share_group(u, v) else 1.0

            for day_idx in range(n_days):
                u_solos = solo_index.get(u, {}).get(day_idx, [])
                v_solos = solo_index.get(v, {}).get(day_idx, [])
                if not u_solos or not v_solos:
                    continue

                # Find best (longest) overlap
                best_start, best_end = 0.0, 0.0
                best_len = 0.0
                for us, ue in u_solos:
                    for vs, ve in v_solos:
                        o_start = max(us, vs)
                        o_end = min(ue, ve)
                        o_len = o_end - o_start
                        if o_len > best_len:
                            best_len = o_len
                            best_start, best_end = o_start, o_end

                if best_len < min_overlap:
                    continue

                load_u = agent_load.get(u, 0) + 1
                load_v = agent_load.get(v, 0) + 1
                score = edge_weight * shared_group / (load_u + load_v)
                # Add small random jitter to break ties
                score += float(self.rng.uniform(0, 0.001))
                candidates.append((score, u, v, day_idx, best_start, best_end))

        log.info("Phase D: %d candidates from %d graph edges", len(candidates), self.graph.number_of_edges())

        # 4. Sort by score descending, greedily select
        candidates.sort(key=lambda c: c[0], reverse=True)

        # Track booked intervals per agent: {agent_id: [(start, end), ...]}
        booked: Dict[str, List[Tuple[float, float]]] = {}
        remote_day_counts: Dict[int, int] = {}
        specs: List[ConversationSpec] = []

        conv_duration = day_length * 0.04  # ~1 hour in a 24h day

        for score, u, v, day_idx, o_start, o_end in candidates:
            if len(specs) >= max_remote:
                break

            # Pick a random offset within the overlap for the conversation
            usable = o_end - o_start - conv_duration
            if usable <= 0:
                c_start = o_start
                c_end = o_end
            else:
                offset = float(self.rng.uniform(0, usable))
                c_start = o_start + offset
                c_end = c_start + conv_duration

            # Check double-booking
            def _conflicts(agent: str, s: float, e: float) -> bool:
                for bs, be in booked.get(agent, []):
                    if s < be and e > bs:
                        return True
                return False

            if _conflicts(u, c_start, c_end) or _conflicts(v, c_start, c_end):
                continue

            # Select modality: 65% text_message, 35% voice_message
            modality = "voice_message" if self.rng.random() < 0.35 else "text_message"

            specs.append(ConversationSpec(
                spec_id=f"spec_rmt_{uuid.uuid4().hex[:8]}",
                participants=sorted([u, v]),
                location_id="",
                start_time=c_start,
                end_time=c_end,
                trigger="remote",
                modality=modality,
                is_group=False,
            ))

            # Update bookkeeping
            booked.setdefault(u, []).append((c_start, c_end))
            booked.setdefault(v, []).append((c_start, c_end))
            remote_day_counts[day_idx] = remote_day_counts.get(day_idx, 0) + 1
            agent_load[u] = agent_load.get(u, 0) + 1
            agent_load[v] = agent_load.get(v, 0) + 1

        return specs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encounter_probability(
        self,
        agent_a: str,
        agent_b: str,
        location_id: str,
    ) -> float:
        """Compute the probability of a casual encounter becoming a conversation."""
        p = self.base_encounter_prob

        # Location type modifier
        loc_type = self._location_type(location_id)
        p *= _LOCATION_TYPE_MODIFIER.get(loc_type, 0.5)

        # Familiarity bonus (Dunbar layer)
        familiarity = self._familiarity(agent_a, agent_b)
        p *= familiarity

        # Shared group bonus
        if self._share_group(agent_a, agent_b):
            p += _SHARED_GROUP_BONUS

        return min(p, 1.0)

    def _familiarity(self, agent_a: str, agent_b: str) -> float:
        """Dunbar-layer-based familiarity multiplier."""
        a = self.agents.get(agent_a)
        b = self.agents.get(agent_b)
        if a is None or b is None:
            return _STRANGER_FAMILIARITY

        # Check if they're social neighbors
        weight_ab = a.social_neighbors.get(agent_b, 0.0)
        weight_ba = b.social_neighbors.get(agent_a, 0.0)
        weight = max(weight_ab, weight_ba)

        if weight <= 0:
            return _STRANGER_FAMILIARITY

        # Map edge weight to Dunbar layer (higher weight = closer)
        if weight >= 0.8:
            return _DUNBAR_FAMILIARITY_BONUS[1]
        elif weight >= 0.5:
            return _DUNBAR_FAMILIARITY_BONUS[2]
        elif weight >= 0.3:
            return _DUNBAR_FAMILIARITY_BONUS[3]
        else:
            return _DUNBAR_FAMILIARITY_BONUS[4]

    def _share_group(self, agent_a: str, agent_b: str) -> bool:
        """Check if two agents share at least one group."""
        groups_a = self._agent_groups.get(agent_a, set())
        groups_b = self._agent_groups.get(agent_b, set())
        return bool(groups_a & groups_b)

    def _location_type(self, location_id: str) -> str:
        """Look up location type from the location map."""
        loc = self._location_map.get(location_id)
        return loc.location_type if loc else "unknown"

    def _roll_modality(self) -> str:
        """Sample a session modality from the weight distribution."""
        modalities = list(self.modality_weights.keys())
        probs = np.array([self.modality_weights[m] for m in modalities], dtype=float)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(len(modalities)) / len(modalities)
        return self.rng.choice(modalities, p=probs)
