"""ScheduleFactory: build a chronological ActivityEntry log for every agent.

Two-phase construction:
  1. build_skeleton()  — deterministic day skeleton (sleep/transit/work/solo),
     no sessions, no memory_text.  Returns skeleton_logs + encounter_windows.
  2. augment_with_sessions() — inserts dialogue/group_meeting entries, splits
     overlapping solo blocks, then batch-generates memory_text and builds
     DailySchedule summaries.

Transit entries are inserted when an agent moves between locations.
"""

from __future__ import annotations

import dataclasses
import uuid
from typing import Dict, List, Optional, Tuple

import json
import re

from MASim.core.agent import EntityAgent
from MASim.core.memory_writer import MemoryWriter
from MASim.core.schema import (
    ActivityEntry, DailySchedule, EncounterWindow, Group, Location,
    PersonaCard, ScheduleSlot, Session,
)
from MASim.prompts import (
    SCHEDULE_ENRICH_SYSTEM,
    SCHEDULE_ENRICH_USER,
)
from MASim.utils.logging import get_logger

log = get_logger(__name__)

# Minimum simulation-time gap to warrant inserting a non-trivial solo entry.
_MIN_SOLO_GAP_FRAC = 0.04   # 4% of total time range

# Fraction of gap to allocate to transit when location changes
_TRANSIT_FRAC = 0.25

# Minimum overlap fraction to count as an encounter window
_MIN_ENCOUNTER_FRAC = 0.02  # 2% of total time range

# Solo activity context strings keyed by location type and time fraction
_SOLO_DESCRIPTIONS = {
    "home_private": [
        ("before work", "morning routine — coffee, getting ready for the day"),
        ("midday",      "at home for lunch, catching up on small chores"),
        ("evening",     "unwinding at home — reading, cooking, or just switching off"),
        ("default",     "some quiet time at home"),
    ],
    "home_shared": [
        ("default",     "time at home"),
    ],
    "workplace": [
        ("default",     "working — focused on tasks, emails, and the usual flow of the day"),
    ],
    "third_place_regular": [
        ("morning",     "morning stop — coffee and a moment to settle before the day starts"),
        ("midday",      "lunch break away from the desk"),
        ("evening",     "winding down over a drink after work"),
        ("default",     "spending some time here"),
    ],
    "third_place_event": [
        ("default",     "time at the venue — part social, part just being out"),
    ],
    "outdoor": [
        ("morning",     "early walk to clear the head"),
        ("default",     "some time outside"),
    ],
    "service": [
        ("default",     "running a quick errand"),
    ],
}


def _clamp_entries(entries: List[ActivityEntry]) -> List[ActivityEntry]:
    """Enforce strict non-overlapping order on a list of ActivityEntries.

    Sorts by start_time, then walks forward ensuring each entry starts no
    earlier than the previous entry's end_time.  Entries whose end_time has
    fallen behind their (adjusted) start_time are snapped to zero-duration.
    """
    entries = sorted(entries, key=lambda e: e.start_time)
    for i in range(1, len(entries)):
        prev = entries[i - 1]
        curr = entries[i]
        if curr.start_time < prev.end_time:
            curr.start_time = prev.end_time
        if curr.end_time < curr.start_time:
            curr.end_time = curr.start_time
    return entries


def _parse_work_schedule(schedule_str: str) -> Tuple[Optional[float], Optional[float], bool]:
    """Parse a human-readable work schedule string into (start_hour, end_hour, weekday_only).

    Returns (None, None, False) for non-fixed schedules (retired, freelance, etc.).
    Default fallback: (9.0, 17.0, True).
    """
    if not schedule_str:
        return (None, None, False)
    s = schedule_str.strip().lower()

    # Non-fixed schedules
    non_fixed = ("retired", "freelance", "freelance irregular", "stay-at-home parent")
    if s in non_fixed:
        return (None, None, False)

    weekday_only = True
    if "weekday" in s:
        weekday_only = True
        s = s.replace("weekdays", "").replace("weekday", "").strip()

    # Check for shift work (not weekday-only)
    if "shift" in s:
        weekday_only = False
        s = re.sub(r"shift\s*work\s*", "", s).strip()

    # Try to parse time range patterns like "6am-2pm", "9-5", "7pm-3am", "9am-5pm"
    pattern = re.compile(
        r"(\d{1,2})\s*(am|pm)?\s*[-–to]+\s*(\d{1,2})\s*(am|pm)?",
        re.IGNORECASE,
    )
    m = pattern.search(s)
    if m:
        start_h = int(m.group(1))
        start_ampm = (m.group(2) or "").lower()
        end_h = int(m.group(3))
        end_ampm = (m.group(4) or "").lower()

        # Convert to 24-hour
        if start_ampm == "pm" and start_h != 12:
            start_h += 12
        elif start_ampm == "am" and start_h == 12:
            start_h = 0

        if end_ampm == "pm" and end_h != 12:
            end_h += 12
        elif end_ampm == "am" and end_h == 12:
            end_h = 0

        # If no am/pm given and it looks like "9-5", assume 9am-5pm
        if not start_ampm and not end_ampm:
            if start_h >= end_h and end_h <= 12:
                # e.g. 9-5 -> 9am-5pm (end is PM)
                end_h += 12 if end_h != 12 else 0

        return (float(start_h), float(end_h), weekday_only)

    # Fallback: standard 9-5 weekdays
    return (9.0, 17.0, True)


def _solo_description(loc_type: str, fraction: float) -> str:
    """Pick the most contextually appropriate solo description."""
    templates = _SOLO_DESCRIPTIONS.get(loc_type, [("default", "some time here")])
    if fraction < 0.25:
        label = "morning"
    elif fraction < 0.55:
        label = "midday"
    elif fraction < 0.80:
        label = "evening"
    else:
        label = "evening"

    for key, desc in templates:
        if key == label:
            return desc
    # Fall back to "default" entry
    for key, desc in templates:
        if key == "default":
            return desc
    return templates[0][1]


def _pick_gap_location(
    persona: PersonaCard,
    locations: List[Location],
    fraction: float,
    current_location_id: str,
) -> Location:
    """Pick a plausible location for a gap activity based on time of day."""
    home_id = persona.home_location_id

    # Morning (<25%): home
    if fraction < 0.25:
        home = next((l for l in locations if l.location_id == home_id), None)
        if home:
            return home

    # Work hours (25%–75%): workplace if agent has a regular schedule
    if 0.25 <= fraction < 0.75 and persona.work_schedule not in ("retired", "freelance irregular", "stay-at-home parent"):
        workplace = next((l for l in locations if l.location_type == "workplace"), None)
        if workplace:
            return workplace

    # Evening (>75%) or freelance/retired: home or third place
    # Prefer third_place_regular if it's not already the current location
    third = next(
        (l for l in locations
         if l.location_type == "third_place_regular" and l.location_id != current_location_id),
        None,
    )
    if fraction > 0.65 and third:
        return third

    # Default: home
    home = next((l for l in locations if l.location_id == home_id), None)
    if home:
        return home

    return locations[0] if locations else Location(location_id="loc_unknown", name="home", location_type="home_private")


def _insert_session_entry(
    entries: List[ActivityEntry],
    new_entry: ActivityEntry,
) -> List[ActivityEntry]:
    """Insert new_entry into entries, splitting any overlapping solo/errand blocks.

    Sleep, transit, dialogue, and group_meeting entries are left unchanged
    (clamping resolves any residual overlaps).  Solo and errand entries that
    overlap the new entry are split: the before-part is kept if it has
    non-zero duration, and the after-part likewise.

    Returns an unsorted list — caller must pass through _clamp_entries().
    """
    result: List[ActivityEntry] = []
    for e in entries:
        # No overlap — keep as-is
        if e.end_time <= new_entry.start_time or e.start_time >= new_entry.end_time:
            result.append(e)
            continue

        # Entries we never split
        if e.activity_type in ("sleep", "transit", "dialogue", "group_meeting"):
            result.append(e)
            continue

        # Solo / errand: split around new_entry
        if e.start_time < new_entry.start_time:
            before = dataclasses.replace(
                e,
                entry_id=f"{e.entry_id}_before",
                end_time=new_entry.start_time,
            )
            result.append(before)
        if e.end_time > new_entry.end_time:
            after = dataclasses.replace(
                e,
                entry_id=f"{e.entry_id}_after",
                start_time=new_entry.end_time,
            )
            result.append(after)
        # The overlapping portion is consumed by new_entry; drop it

    result.append(new_entry)
    return result


class ScheduleFactory:
    """Build ActivityEntry logs and DailySchedule summaries for all agents."""

    def __init__(
        self,
        memory_writer: MemoryWriter,
        transit_matrix: Dict[str, Dict[str, int]],
    ):
        self.memory_writer = memory_writer
        self.transit_matrix = transit_matrix

    # ------------------------------------------------------------------
    # Phase A: Skeleton (no sessions, no memory_text)
    # ------------------------------------------------------------------

    def build_skeleton(
        self,
        agents: Dict[str, EntityAgent],
        locations: List[Location],
        time_range: Tuple[float, float],
        groups: Optional[List[Group]] = None,
        llm_client: Optional[object] = None,
        dry_run: bool = False,
    ) -> Tuple[Dict[str, List[ActivityEntry]], List[EncounterWindow]]:
        """Build deterministic day skeletons and detect encounter windows.

        Returns (skeleton_logs, encounter_windows) where skeleton_logs is
        keyed by agent_id and contains ActivityEntries without memory_text,
        and encounter_windows is the list of co-location overlaps.

        If ``groups`` is provided, group meeting entries are inserted into
        each member's skeleton before encounter detection, ensuring that
        group meetings appear as encounter windows.

        If ``llm_client`` is provided and ``dry_run`` is False, solo/transit
        activity descriptions are enriched via LLM to be persona-specific.
        """
        location_map = {l.location_id: l for l in locations}
        skeleton_logs: Dict[str, List[ActivityEntry]] = {}
        for agent_id, agent in agents.items():
            skeleton_logs[agent_id] = self._build_agent_skeleton(
                agent, locations, location_map, time_range,
            )

        # Insert group meeting entries into member skeletons
        if groups:
            n_meetings = self._insert_group_meetings(
                skeleton_logs, groups, time_range,
            )
            log.info("Inserted %d group meeting entries across all agents", n_meetings)

        # Detect and resolve role conflicts (overlapping group meetings)
        if groups:
            n_conflicts = self.detect_role_conflicts(skeleton_logs, groups)
            if n_conflicts:
                log.info("Detected and resolved %d role conflicts", n_conflicts)

        encounter_windows = _detect_encounters(skeleton_logs, time_range)

        # LLM enrichment: rewrite generic descriptions to be persona-specific
        if llm_client is not None and not dry_run:
            location_map = {l.location_id: l for l in locations}
            self._enrich_skeleton_descriptions(
                skeleton_logs, agents, location_map, llm_client, time_range,
            )

        log.info(
            "ScheduleFactory.build_skeleton: %d agents, %d encounter windows",
            len(agents), len(encounter_windows),
        )
        return skeleton_logs, encounter_windows

    # ------------------------------------------------------------------
    # LLM enrichment of skeleton descriptions
    # ------------------------------------------------------------------

    def _enrich_skeleton_descriptions(
        self,
        skeleton_logs: Dict[str, List[ActivityEntry]],
        agents: Dict[str, EntityAgent],
        location_map: Dict[str, "Location"],
        llm_client: object,
        time_range: Tuple[float, float],
    ) -> None:
        """Rewrite solo/transit descriptions to be persona-specific via LLM.

        Modifies skeleton_logs in place. If an LLM call fails or returns the
        wrong number of descriptions, the original generic text is kept.
        """
        sim_start, sim_end = time_range
        n_days = max(1, int(sim_end - sim_start))

        tasks = []         # LLM batch tasks
        task_meta = []     # (agent_id, list of entry indices into skeleton_logs[agent_id])

        for agent_id, agent in agents.items():
            entries = skeleton_logs[agent_id]
            p = agent.persona

            # Group enrichable entries by day to keep each prompt small
            for day in range(n_days):
                day_start = sim_start + day
                day_end = day_start + 1.0
                enrichable = []
                activity_lines = []

                for idx, entry in enumerate(entries):
                    if entry.activity_type not in ("solo", "transit"):
                        continue
                    if entry.start_time < day_start or entry.start_time >= day_end:
                        continue
                    enrichable.append(idx)
                    day_frac = entry.start_time - day_start
                    hour = day_frac * 24
                    if hour < 10:
                        time_label = "morning"
                    elif hour < 14:
                        time_label = "midday"
                    elif hour < 18:
                        time_label = "afternoon"
                    else:
                        time_label = "evening"
                    loc = location_map.get(entry.location_id)
                    loc_name = loc.name if loc else entry.location_id
                    activity_lines.append(
                        f'{len(activity_lines)+1}. [{time_label}, {loc_name}] "{entry.description}"'
                    )

                if not enrichable:
                    continue

                day_label = "weekday" if (day % 7) < 5 else "weekend"
                prompt = SCHEDULE_ENRICH_USER.format(
                    name=p.name,
                    age=p.age,
                    occupation=p.occupation,
                    hobbies=", ".join(p.hobbies) if p.hobbies else "none listed",
                    daily_routine=p.daily_routine_notes or "not specified",
                    concerns=", ".join(p.current_concerns) if p.current_concerns else "none listed",
                    speaking_style=p.speaking_style or "neutral",
                    activities="\n".join(activity_lines),
                    n=len(activity_lines),
                )
                tasks.append({
                    "system": SCHEDULE_ENRICH_SYSTEM,
                    "user": prompt,
                    "max_tokens": 2048,
                    "tags": {"phase": "schedule_enrich"},
                })
                task_meta.append((agent_id, enrichable))

        if not tasks:
            return

        log.info("Enriching skeleton descriptions: %d agent-day batches via LLM...", len(tasks))
        responses = llm_client.generate_batch(tasks)

        n_enriched = 0
        n_failed = 0
        for resp, (agent_id, entry_indices) in zip(responses, task_meta):
            try:
                text = resp.strip()
                # Strip think tags and markdown fences
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                text = re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = re.sub(r"\n?```\s*$", "", text)
                start = text.find("[")
                end = text.rfind("]")
                if start == -1 or end == -1:
                    raise ValueError("No JSON array found")
                descriptions = json.loads(text[start:end+1])
                if len(descriptions) != len(entry_indices):
                    raise ValueError(
                        f"Expected {len(entry_indices)} descriptions, got {len(descriptions)}"
                    )
                entries = skeleton_logs[agent_id]
                for idx, new_desc in zip(entry_indices, descriptions):
                    if isinstance(new_desc, str) and new_desc.strip():
                        entries[idx].description = new_desc.strip()
                n_enriched += 1
            except Exception as e:
                log.warning("Failed to enrich skeleton for %s: %s", agent_id, e)
                n_failed += 1

        log.info(
            "Skeleton enrichment: %d/%d agents enriched, %d failed",
            n_enriched, len(tasks), n_failed,
        )

    # ------------------------------------------------------------------
    # Group meeting insertion (Phase 3 lifecycle)
    # ------------------------------------------------------------------

    def _insert_group_meetings(
        self,
        skeleton_logs: Dict[str, List[ActivityEntry]],
        groups: List[Group],
        time_range: Tuple[float, float],
    ) -> int:
        """Insert group_meeting ActivityEntries into member skeletons.

        For each group with a meeting_schedule, picks specific days based on
        meeting frequency and maps the meeting's start_hour to the correct
        sim time within that day: day + start_hour / 24.0.

        Returns the total number of meeting entries inserted.
        """
        sim_start, sim_end = time_range
        first_day = int(sim_start)
        last_day = int(sim_end)
        total_days = last_day - first_day
        n_inserted = 0

        for group in groups:
            if not group.meeting_schedule:
                continue
            member_ids = list(group.members.keys())
            if len(member_ids) < 2:
                continue

            for m_idx, meeting in enumerate(group.meeting_schedule):
                start_hour = meeting.get("start_hour", 9.0)
                duration_hours = meeting.get("duration_hours", 1.0)
                location_id = meeting.get("location_id", group.home_location_id)
                description = meeting.get("activity_description", "Group meeting")
                # Determine meeting frequency: default every 3 days
                frequency_days = meeting.get("frequency_days", 3)

                # Pick specific days for this meeting
                # Offset by meeting index to spread different meetings across days
                meeting_days = []
                offset = (hash(group.group_id + str(m_idx)) % frequency_days)
                for d in range(first_day + offset, last_day, max(1, frequency_days)):
                    meeting_days.append(d)

                for day in meeting_days:
                    meeting_start = day + start_hour / 24.0
                    meeting_end = day + (start_hour + duration_hours) / 24.0
                    # Clamp to day boundary
                    meeting_end = min(meeting_end, float(day + 1))
                    if meeting_end <= meeting_start:
                        continue

                    for member_id in member_ids:
                        if member_id not in skeleton_logs:
                            continue
                        other_members = [m for m in member_ids if m != member_id]
                        new_entry = ActivityEntry(
                            entry_id=f"act_{member_id}_grpmtg_{uuid.uuid4().hex[:8]}",
                            agent_id=member_id,
                            activity_type="group_meeting",
                            start_time=meeting_start,
                            end_time=meeting_end,
                            location_id=location_id,
                            group_id=group.group_id,
                            participants=other_members,
                            description=f"{description} ({group.name})",
                        )
                        skeleton_logs[member_id] = _insert_session_entry(
                            skeleton_logs[member_id], new_entry,
                        )
                        skeleton_logs[member_id] = _clamp_entries(skeleton_logs[member_id])
                        n_inserted += 1

        return n_inserted

    # ------------------------------------------------------------------
    # Role conflict detection (Phase 4)
    # ------------------------------------------------------------------

    _ROLE_PRIORITY: Dict[str, int] = {
        "family": 4,
        "work_team": 3,
        "hobby": 2,
        "neighbourhood": 1,
        "civic": 1,
        "friendship_circle": 1,
    }

    def detect_role_conflicts(
        self,
        skeleton_logs: Dict[str, List[ActivityEntry]],
        groups: List[Group],
    ) -> int:
        """Detect overlapping group_meeting entries for the same agent.

        For each agent, find pairs of group_meeting entries with overlapping
        time.  Keep the higher-priority meeting; replace the lower-priority
        one with a ``role_conflict`` ActivityEntry.  Ties: keep the earlier
        one.

        Returns the total number of conflicts resolved.
        """
        group_map = {g.group_id: g for g in groups}
        n_conflicts = 0

        for agent_id, entries in skeleton_logs.items():
            meetings = [
                e for e in entries if e.activity_type == "group_meeting"
            ]
            if len(meetings) < 2:
                continue

            # Sort by start_time for pairwise sweep
            meetings.sort(key=lambda e: e.start_time)
            to_replace: Dict[str, ActivityEntry] = {}  # entry_id -> replacement

            for i in range(len(meetings)):
                for j in range(i + 1, len(meetings)):
                    a = meetings[i]
                    b = meetings[j]
                    if b.start_time >= a.end_time:
                        break  # no further overlaps with a (sorted)
                    # Overlap detected
                    if a.entry_id in to_replace or b.entry_id in to_replace:
                        continue  # already resolved

                    grp_a = group_map.get(a.group_id)
                    grp_b = group_map.get(b.group_id)
                    prio_a = self._ROLE_PRIORITY.get(
                        grp_a.group_type if grp_a else "", 1
                    )
                    prio_b = self._ROLE_PRIORITY.get(
                        grp_b.group_type if grp_b else "", 1
                    )

                    # Higher priority wins; ties: keep the earlier one (a)
                    if prio_b > prio_a:
                        winner, loser = b, a
                        winner_grp, loser_grp = grp_b, grp_a
                    else:
                        winner, loser = a, b
                        winner_grp, loser_grp = grp_a, grp_b

                    replacement = ActivityEntry(
                        entry_id=f"{loser.entry_id}_conflict",
                        agent_id=agent_id,
                        activity_type="role_conflict",
                        start_time=loser.start_time,
                        end_time=loser.end_time,
                        location_id=loser.location_id,
                        group_id=loser.group_id,
                        description=(
                            f"Scheduling conflict — missed {loser_grp.name if loser_grp else loser.group_id}"
                            f" because of {winner_grp.name if winner_grp else winner.group_id}."
                        ),
                        metadata={
                            "conflict_type": "role_conflict",
                            "winner_group_id": winner.group_id,
                            "loser_group_id": loser.group_id,
                            "winner_group_name": winner_grp.name if winner_grp else winner.group_id,
                            "loser_group_name": loser_grp.name if loser_grp else loser.group_id,
                        },
                    )
                    to_replace[loser.entry_id] = replacement
                    n_conflicts += 1

            if to_replace:
                skeleton_logs[agent_id] = [
                    to_replace[e.entry_id] if e.entry_id in to_replace else e
                    for e in entries
                ]

        return n_conflicts

    # ------------------------------------------------------------------
    # Phase B: Augment skeleton with sessions
    # ------------------------------------------------------------------

    def augment_with_sessions(
        self,
        skeleton_logs: Dict[str, List[ActivityEntry]],
        sessions: List[Session],
        agents: Dict[str, EntityAgent],
        locations: List[Location],
        time_range: Tuple[float, float],
        dry_run: bool = False,
    ) -> Tuple[Dict[str, List[ActivityEntry]], Dict[str, DailySchedule]]:
        """Insert session entries into skeleton logs, generate memory_text, build schedules.

        Returns (full_logs, schedules) keyed by agent_id.
        """
        location_map = {l.location_id: l for l in locations}
        persona_map = {aid: a.persona for aid, a in agents.items()}
        agent_display_names = {aid: a.persona.name for aid, a in agents.items()}
        sim_start, sim_end = time_range
        sim_duration = max(sim_end - sim_start, 1.0)
        min_gap = sim_duration * _MIN_SOLO_GAP_FRAC

        # Deep-copy the skeleton so we don't mutate the originals
        all_entries: Dict[str, List[ActivityEntry]] = {
            aid: list(entries) for aid, entries in skeleton_logs.items()
        }

        # Group sessions by agent
        sessions_by_agent: Dict[str, List[Session]] = {aid: [] for aid in agents}
        for sess in sessions:
            for pid in sess.participants:
                if pid in sessions_by_agent:
                    sessions_by_agent[pid].append(sess)

        # Insert session entries per agent
        for agent_id, agent in agents.items():
            my_sessions = sorted(sessions_by_agent[agent_id], key=lambda s: s.start_time)
            entries = all_entries[agent_id]

            for sess in my_sessions:
                is_group = sess.metadata.get("group_conversation", False) or len(sess.participants) > 2
                activity_type = "group_meeting" if is_group else "dialogue"
                other_participants = [p for p in sess.participants if p != agent_id]

                fact_hint = ""
                for turn in sess.turns:
                    if turn.speaker_id != agent_id and turn.extracted_facts:
                        fact_hint = turn.extracted_facts[0][:80]
                        break

                sess_loc_id = sess.location_id or agent.persona.home_location_id or ""
                loc_name = location_map.get(sess_loc_id, Location()).name or "somewhere"
                other_names = [
                    agent_display_names.get(p, p.split("_")[0].capitalize())
                    for p in other_participants
                ]
                names_str = ", ".join(other_names) if other_names else "others"

                sess_end = max(sess.end_time, sess.start_time + min_gap)
                new_entry = ActivityEntry(
                    entry_id=f"act_{agent_id}_{uuid.uuid4().hex[:8]}",
                    agent_id=agent_id,
                    activity_type=activity_type,
                    start_time=sess.start_time,
                    end_time=sess_end,
                    location_id=sess_loc_id,
                    group_id=sess.group_id,
                    role_name=sess.role_name,
                    participants=other_participants,
                    description=f"Conversation with {names_str} at {loc_name}.",
                    linked_session_id=sess.session_id,
                    metadata={"fact_hint": fact_hint, "participant_names": agent_display_names},
                )
                entries = _insert_session_entry(entries, new_entry)

            all_entries[agent_id] = _clamp_entries(entries)

        # Batch-generate memory_text
        flat_entries = [e for entries in all_entries.values() for e in entries]
        self.memory_writer.fill_memories(
            flat_entries, persona_map, location_map, time_range, dry_run=dry_run,
        )

        # Update agent knowledge states
        for agent_id, entries in all_entries.items():
            agent = agents[agent_id]
            for entry in entries:
                agent.knowledge.update_from_activity(entry)

        # Build DailySchedule summaries
        schedules: Dict[str, DailySchedule] = {}
        for agent_id, entries in all_entries.items():
            schedules[agent_id] = self._build_schedule(
                agent_id, agents[agent_id].persona, entries, time_range,
            )

        total_entries = sum(len(v) for v in all_entries.values())
        log.info(
            "ScheduleFactory.augment_with_sessions: %d entries across %d agents",
            total_entries, len(agents),
        )
        return all_entries, schedules

    # ------------------------------------------------------------------
    # Per-agent skeleton construction
    # ------------------------------------------------------------------

    def _build_agent_skeleton(
        self,
        agent: EntityAgent,
        locations: List[Location],
        location_map: Dict[str, Location],
        time_range: Tuple[float, float],
    ) -> List[ActivityEntry]:
        """Build deterministic skeleton ActivityEntries for one agent across all days.

        Each integer day d in time_range spans sim time [d, d+1].
        Within each day, persona clock hours map to sim time via:
            to_sim(hour) = d + hour / 24.0

        No sessions, no memory_text.
        """
        sim_start, sim_end = time_range
        persona = agent.persona
        entries: List[ActivityEntry] = []

        home_id = persona.home_location_id or (locations[0].location_id if locations else "")

        # Parse persona schedule fields
        wake_hour = persona.sleep_end_hour if persona.sleep_end_hour else 7.0
        bed_hour = persona.sleep_start_hour if persona.sleep_start_hour else 23.0
        work_start_h, work_end_h, weekday_only = _parse_work_schedule(
            persona.work_schedule or ""
        )
        has_fixed_work = work_start_h is not None

        # Find key locations
        workplace = next((l for l in locations if l.location_type == "workplace"), None)
        third_place = next(
            (l for l in locations
             if l.location_type == "third_place_regular" and l.location_id != home_id),
            None,
        )
        hobby_loc = next(
            (l for l in locations
             if l.location_type in ("third_place_event", "outdoor") and l.location_id != home_id),
            None,
        )

        # Sim-time duration of 30 min within one day
        transit_dur = 0.5 / 24.0
        # Sim-time duration of 45 min (lunch)
        lunch_dur = 0.75 / 24.0

        first_day = int(sim_start)
        last_day = int(sim_end)  # exclusive: for [0,15] iterate 0..14

        for d in range(first_day, last_day):
            # Per-agent-per-day jitter: ±10 minutes in sim time
            jitter_minutes = (hash(agent.agent_id + str(d)) % 20) - 10
            jitter = jitter_minutes / (24.0 * 60.0)

            def to_sim(hour: float) -> float:
                return d + hour / 24.0 + jitter

            wake_sim = to_sim(wake_hour)
            bed_sim = to_sim(bed_hour)
            day_start = float(d)
            day_end = float(d + 1)

            # Clamp to day boundaries
            wake_sim = max(day_start, min(day_end, wake_sim))
            bed_sim = max(wake_sim, min(day_end, bed_sim))

            # Determine if this is a workday
            day_of_week = d % 7  # 0=Mon .. 6=Sun
            is_weekend = day_of_week >= 5
            is_workday = has_fixed_work and workplace and (not weekday_only or not is_weekend)

            if is_workday:
                # Workday schedule
                work_start_sim = to_sim(work_start_h)
                work_end_sim = to_sim(work_end_h)
                # Handle overnight shifts (e.g. 7pm-3am): skip for now, treat as next-day end
                if work_end_sim <= work_start_sim:
                    work_end_sim = to_sim(work_end_h + 24.0)
                    work_end_sim = min(work_end_sim, day_end)
                work_start_sim = max(wake_sim, min(day_end, work_start_sim))
                work_end_sim = max(work_start_sim, min(day_end, work_end_sim))

                # Lunch at midpoint of work hours
                lunch_mid_h = (work_start_h + work_end_h) / 2.0
                if work_end_h < work_start_h:
                    lunch_mid_h = (work_start_h + work_end_h + 24.0) / 2.0
                lunch_start_sim = to_sim(lunch_mid_h) - lunch_dur / 2.0
                lunch_end_sim = lunch_start_sim + lunch_dur
                lunch_start_sim = max(work_start_sim, min(work_end_sim, lunch_start_sim))
                lunch_end_sim = max(lunch_start_sim, min(work_end_sim, lunch_end_sim))

                # Transit times
                transit_to_work_start = work_start_sim - transit_dur
                transit_to_work_start = max(wake_sim, transit_to_work_start)
                transit_home_start = work_end_sim
                transit_home_end = work_end_sim + transit_dur
                transit_home_end = min(bed_sim, transit_home_end)

                blocks = [
                    # Opening sleep
                    ("sleep", day_start, wake_sim, home_id, "start"),
                    # Morning solo at home
                    ("solo", wake_sim, transit_to_work_start, home_id, "home_private"),
                    # Transit home -> work
                    ("transit", transit_to_work_start, work_start_sim, home_id, workplace.location_id),
                    # Work block 1
                    ("solo", work_start_sim, lunch_start_sim, workplace.location_id, "workplace"),
                    # Lunch
                    ("solo", lunch_start_sim, lunch_end_sim,
                     third_place.location_id if third_place else workplace.location_id,
                     (third_place.location_type if third_place else "workplace")),
                    # Work block 2
                    ("solo", lunch_end_sim, work_end_sim, workplace.location_id, "workplace"),
                    # Transit work -> home
                    ("transit", transit_home_start, transit_home_end, workplace.location_id, home_id),
                    # Evening solo at home
                    ("solo", transit_home_end, bed_sim, home_id, "home_private"),
                    # Closing sleep
                    ("sleep", bed_sim, day_end, home_id, "end"),
                ]
            else:
                # Weekend / non-worker day
                noon_sim = to_sim(12.0)
                noon_sim = max(wake_sim, min(bed_sim, noon_sim))
                lunch_end_sim = noon_sim + lunch_dur
                lunch_end_sim = max(noon_sim, min(bed_sim, lunch_end_sim))

                afternoon_loc = hobby_loc or third_place
                afternoon_loc_id = afternoon_loc.location_id if afternoon_loc else home_id
                afternoon_loc_type = afternoon_loc.location_type if afternoon_loc else "home_private"

                blocks = [
                    # Opening sleep
                    ("sleep", day_start, wake_sim, home_id, "start"),
                    # Morning solo
                    ("solo", wake_sim, noon_sim, home_id, "home_private"),
                    # Lunch at third place
                    ("solo", noon_sim, lunch_end_sim,
                     third_place.location_id if third_place else home_id,
                     (third_place.location_type if third_place else "home_private")),
                    # Afternoon/evening solo (varied location)
                    ("solo", lunch_end_sim, bed_sim, afternoon_loc_id, afternoon_loc_type),
                    # Closing sleep
                    ("sleep", bed_sim, day_end, home_id, "end"),
                ]

            # Emit entries from blocks
            for block in blocks:
                btype = block[0]
                bstart = block[1]
                bend = block[2]

                # Skip zero or negative duration
                if bend - bstart < 1e-6:
                    continue

                if btype == "sleep":
                    label = block[4]  # "start" or "end"
                    entries.append(self._sleep_entry(
                        agent_id=agent.agent_id,
                        start_time=bstart,
                        end_time=bend,
                        location_id=block[3],
                        persona=persona,
                        label=label,
                    ))
                elif btype == "transit":
                    from_loc_id = block[3]
                    to_loc_id = block[4]
                    entries.append(self._transit_entry(
                        agent_id=agent.agent_id,
                        start_time=bstart,
                        end_time=bend,
                        from_loc_id=from_loc_id,
                        to_loc_id=to_loc_id,
                        location_map=location_map,
                    ))
                else:
                    # solo
                    loc_id = block[3]
                    loc_type = block[4]
                    # Compute fraction within day for description selection
                    day_frac = (bstart - day_start)
                    entries.append(ActivityEntry(
                        entry_id=f"act_{agent.agent_id}_{uuid.uuid4().hex[:8]}",
                        agent_id=agent.agent_id,
                        activity_type="solo",
                        start_time=bstart,
                        end_time=bend,
                        location_id=loc_id,
                        description=_solo_description(loc_type, fraction=day_frac),
                    ))

        return _clamp_entries(entries)

    # ------------------------------------------------------------------
    # Entry constructors
    # ------------------------------------------------------------------

    def _sleep_entry(
        self,
        agent_id: str,
        start_time: float,
        end_time: float,
        location_id: str,
        persona: PersonaCard,
        label: str = "start",
    ) -> ActivityEntry:
        is_start = label == "start"
        desc = (
            f"Sleeping — {persona.sleep_start_hour:.0f}:00 to {persona.sleep_end_hour:.0f}:00."
            if is_start else
            f"End of day, winding down for sleep."
        )
        return ActivityEntry(
            entry_id=f"act_{agent_id}_sleep_{label}_{uuid.uuid4().hex[:6]}",
            agent_id=agent_id,
            activity_type="sleep",
            start_time=start_time,
            end_time=end_time,
            location_id=location_id,
            description=desc,
        )

    def _transit_entry(
        self,
        agent_id: str,
        start_time: float,
        end_time: float,
        from_loc_id: str,
        to_loc_id: str,
        location_map: Dict[str, Location],
    ) -> ActivityEntry:
        from_loc = location_map.get(from_loc_id)
        to_loc = location_map.get(to_loc_id)
        minutes = self.transit_matrix.get(from_loc_id, {}).get(to_loc_id, 15)
        from_name = from_loc.name if from_loc else "previous location"
        to_name = to_loc.name if to_loc else "next location"
        return ActivityEntry(
            entry_id=f"act_{agent_id}_transit_{uuid.uuid4().hex[:6]}",
            agent_id=agent_id,
            activity_type="transit",
            start_time=start_time,
            end_time=end_time,
            location_id=to_loc_id,
            description=f"Traveling from {from_name} to {to_name}.",
            metadata={
                "from_location_id": from_loc_id,
                "to_location_id": to_loc_id,
                "transit_minutes": minutes,
            },
        )

    # ------------------------------------------------------------------
    # DailySchedule summary builder
    # ------------------------------------------------------------------

    def _build_schedule(
        self,
        agent_id: str,
        persona: PersonaCard,
        entries: List[ActivityEntry],
        time_range: Tuple[float, float],
    ) -> DailySchedule:
        """Convert a list of ActivityEntries into a DailySchedule summary."""
        sim_start, sim_end = time_range
        sim_duration = max(sim_end - sim_start, 1.0)
        wake_h = persona.sleep_end_hour or 7.0
        active_h = ((persona.sleep_start_hour or 23.0) - wake_h) % 24 or 16.0

        def to_hour(sim_t: float) -> float:
            frac = (sim_t - sim_start) / sim_duration
            return wake_h + frac * active_h

        slots = []
        for i, entry in enumerate(entries):
            start_h = to_hour(entry.start_time)
            if entry.end_time > entry.start_time:
                end_h = to_hour(entry.end_time)
            else:
                next_start = to_hour(entries[i + 1].start_time) if i + 1 < len(entries) else start_h + 0.5
                end_h = min(start_h + 0.5, next_start)
                if end_h <= start_h:
                    end_h = start_h
            slots.append(ScheduleSlot(
                start_hour=round(start_h, 2),
                end_hour=round(end_h, 2),
                activity_type=entry.activity_type,
                location_id=entry.location_id,
                group_id=entry.group_id,
                role_name=entry.role_name,
                is_fixed=entry.activity_type in ("sleep", "group_meeting"),
                participants=list(entry.participants),
            ))

        active_roles = list({
            entry.role_name
            for entry in entries
            if entry.role_name
        })

        is_weekday = True  # Phase 3 will add day-type detection
        return DailySchedule(
            agent_id=agent_id,
            day_index=0,
            day_type="weekday" if is_weekday else "weekend",
            slots=slots,
            active_roles=active_roles,
        )


# ---------------------------------------------------------------------------
# Encounter detection (module-level, used by build_skeleton)
# ---------------------------------------------------------------------------

def _detect_encounters(
    skeleton_logs: Dict[str, List[ActivityEntry]],
    time_range: Tuple[float, float],
) -> List[EncounterWindow]:
    """Detect co-location overlaps between agents in skeleton logs.

    Groups all agents present at the same location during overlapping time
    intervals into a single multi-agent window (instead of O(N²) pairwise
    windows).  Excludes sleep and transit entries.

    Returns a list of EncounterWindow objects.
    """
    sim_start, sim_end = time_range
    sim_duration = max(sim_end - sim_start, 1.0)
    # Any co-location overlap >= 0.1 sim-time units counts as an encounter.
    min_overlap = 0.1

    # Build location-presence index: {location_id: [(agent_id, start, end), ...]}
    presence: Dict[str, List[Tuple[str, float, float]]] = {}
    for agent_id, entries in skeleton_logs.items():
        for entry in entries:
            if entry.activity_type in ("sleep", "transit"):
                continue
            loc_id = entry.location_id
            if not loc_id:
                continue
            presence.setdefault(loc_id, []).append(
                (agent_id, entry.start_time, entry.end_time)
            )

    windows: List[EncounterWindow] = []
    win_idx = 0

    for loc_id, presences in presence.items():
        if len(presences) < 2:
            continue

        # Sweep-line: find maximal overlap intervals with 2+ agents.
        # Collect all start/end events, sweep, and emit one window per
        # contiguous interval where the same set of agents overlaps.
        events: List[Tuple[float, int, str]] = []  # (time, +1/-1, agent_id)
        for agent_id, start, end in presences:
            events.append((start, 1, agent_id))
            events.append((end, -1, agent_id))
        # Sort: by time, then ends (-1) before starts (+1) at same time
        events.sort(key=lambda e: (e[0], e[1]))

        active: Dict[str, int] = {}  # agent_id -> ref_count
        prev_time = events[0][0]

        for time, delta, agent_id in events:
            if time > prev_time and len(active) >= 2:
                if time - prev_time >= min_overlap:
                    windows.append(EncounterWindow(
                        window_id=f"enc_{win_idx:06d}",
                        agent_ids=sorted(active.keys()),
                        location_id=loc_id,
                        start_time=prev_time,
                        end_time=time,
                        encounter_type="co_located",
                    ))
                    win_idx += 1
            if delta == 1:
                active[agent_id] = active.get(agent_id, 0) + 1
            else:
                active[agent_id] = active.get(agent_id, 0) - 1
                if active[agent_id] <= 0:
                    del active[agent_id]
            prev_time = time

    return windows


# ---------------------------------------------------------------------------
# Stub helper — kept for backward compatibility with any external callers
# ---------------------------------------------------------------------------

def _build_participant_name_map(
    agent_id: str,
    sessions: List[Session],
    agents: Dict[str, "EntityAgent"],
) -> Dict[str, str]:
    """Return {agent_id: display_name} for all co-participants of this agent."""
    result: Dict[str, str] = {}
    for sess in sessions:
        if agent_id in sess.participants:
            for pid in sess.participants:
                if pid != agent_id and pid in agents:
                    result[pid] = agents[pid].persona.name
    return result
