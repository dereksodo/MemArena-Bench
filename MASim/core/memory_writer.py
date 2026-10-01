"""MemoryWriter: generate first-person narrative memory_text for ActivityEntry objects.

Solo and group_meeting activities use a batched LLM call so every agent's day
can be narrated in a single batch round-trip.  Sleep, transit, errand, and
dialogue activities use fast templates (no LLM call required).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from MASim.core.schema import ActivityEntry, Location, PersonaCard
from MASim.prompts import MEMORY_WRITER_SYSTEM as MEMORY_SYSTEM_PROMPT
from MASim.prompts import MEMORY_WRITER_USER as MEMORY_USER_TEMPLATE
from MASim.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary register lookup (mirrors agent.py)
# ---------------------------------------------------------------------------

_EDU_VOCAB: Dict[str, str] = {
    "high school diploma":
        "short sentences, everyday words, contractions freely",
    "some college or trade school":
        "practical and direct, occasional trade-specific terms",
    "bachelor's degree":
        "fluent and articulate, still conversational",
    "master's degree":
        "precise vocabulary, field-specific terms when relevant",
    "phd or doctoral":
        "may use academic register, tends toward complexity",
}

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

_SLEEP_TEMPLATES = [
    "Slept from {start_clock} to {end_clock} — {hours:.0f} hours. {note}",
    "Got {hours:.0f} hours. {note}",
    "Finally got to sleep. Woke up at {end_clock}. {note}",
]

_SLEEP_NOTES = [
    "Woke up groggy.",
    "Slept well.",
    "Kept waking up through the night.",
    "Out the moment my head hit the pillow.",
    "Dreamt about work, which was annoying.",
    "Needed that.",
    "Could have slept longer.",
]

_TRANSIT_TEMPLATES = [
    "Got from {from_name} to {to_name} — about {minutes} minutes.",
    "Made the {minutes}-minute trip to {to_name}.",
    "Commuted to {to_name}. {minutes} minutes.",
    "Walked over to {to_name}. Took around {minutes} minutes.",
    "Headed out to {to_name} — around {minutes} minutes door to door.",
]

_ERRAND_TEMPLATES = [
    "Ran an errand at {location_name}. Quick in and out.",
    "Stopped by {location_name} to take care of something.",
    "Dealt with a quick errand at {location_name}.",
    "Swung by {location_name}. Nothing major.",
]

_DIALOGUE_TEMPLATES = [
    "Talked with {names} at {location_name} {time_label}. {fact_hint}",
    "Had a conversation with {names} {time_label}, at {location_name}. {fact_hint}",
    "Caught up with {names} at {location_name}. {fact_hint}",
    "Spent some time talking with {names} {time_label}. {fact_hint}",
]

_ROLE_CONFLICT_TEMPLATES = [
    "Had to miss {loser} because of a clash with {winner}. Frustrating.",
    "Scheduling conflict — {loser} overlapped with {winner}. Had to prioritise.",
    "Couldn't make it to {loser} today. {winner} took priority. Felt bad about it.",
]


def _hour_to_clock(h: float) -> str:
    """Convert fractional hour (0–24) to readable clock string like '7am', '11:30pm'."""
    h = h % 24
    period = "am" if h < 12 else "pm"
    display = int(h) % 12 or 12
    mins = int((h % 1) * 60)
    if mins:
        return f"{display}:{mins:02d}{period}"
    return f"{display}{period}"


def _duration_str(hours: float) -> str:
    if hours < 0.25:
        return "a few minutes"
    if hours < 1.0:
        return f"about {int(hours * 60)} minutes"
    if hours < 1.5:
        return "about an hour"
    return f"about {hours:.1f} hours"


def _first_name(name: str) -> str:
    return name.split()[0] if name else "them"


def _rng_pick(lst: list, seed_val: int) -> str:
    return lst[seed_val % len(lst)]


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MemoryWriter:
    """Generate first-person memory_text for ActivityEntry objects.

    Usage pattern:
        writer = MemoryWriter(llm_client)
        entries = writer.fill_memories(entries, agents, location_map, time_range, dry_run)

    ``fill_memories`` mutates entries in-place (setting memory_text) and
    returns the same list for chaining.  All LLM calls are batched.
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def fill_memories(
        self,
        entries: List[ActivityEntry],
        personas: Dict[str, PersonaCard],
        location_map: Dict[str, Location],
        time_range: Tuple[float, float],
        dry_run: bool = False,
    ) -> List[ActivityEntry]:
        """Fill memory_text for all entries in-place.

        LLM is only called for solo / group_meeting entries.
        All other types use deterministic templates.
        """
        llm_indices: List[int] = []
        llm_tasks: List[Dict[str, Any]] = []

        for i, entry in enumerate(entries):
            persona = personas.get(entry.agent_id)
            if persona is None:
                entry.memory_text = entry.description
                continue

            loc = location_map.get(entry.location_id)

            if entry.activity_type == "sleep":
                entry.memory_text = self._sleep_text(entry, persona)

            elif entry.activity_type == "transit":
                entry.memory_text = self._transit_text(entry, location_map)

            elif entry.activity_type == "errand":
                entry.memory_text = self._errand_text(entry, loc)

            elif entry.activity_type == "dialogue":
                entry.memory_text = self._dialogue_text(entry, persona, loc, time_range)

            elif entry.activity_type == "role_conflict":
                entry.memory_text = self._role_conflict_text(entry)

            else:  # solo, group_meeting — use LLM
                if dry_run:
                    entry.memory_text = self._solo_template_fallback(entry, persona, loc, time_range)
                else:
                    task = self._build_llm_task(entry, persona, loc, time_range)
                    llm_tasks.append(task)
                    llm_indices.append(i)

        # Batch LLM call for solo / group_meeting entries
        if llm_tasks:
            log.info("MemoryWriter: generating %d activity memories via LLM...", len(llm_tasks))
            responses = self.llm_client.generate_batch(llm_tasks)
            for idx, resp in zip(llm_indices, responses):
                entries[idx].memory_text = resp.strip() if resp else entries[idx].description

        return entries

    # ------------------------------------------------------------------
    # Template-based generators (no LLM)
    # ------------------------------------------------------------------

    def _sleep_text(self, entry: ActivityEntry, persona: PersonaCard) -> str:
        duration_h = (entry.end_time - entry.start_time)
        # Map sim duration to real hours via persona sleep window
        sleep_window = (persona.sleep_start_hour - persona.sleep_end_hour) % 24 or 7.0
        # Use persona's actual sleep hours for the narrative
        start_clock = _hour_to_clock(persona.sleep_start_hour)
        end_clock = _hour_to_clock(persona.sleep_end_hour)
        seed = hash(entry.entry_id) % 1000
        note = _rng_pick(_SLEEP_NOTES, seed)
        tmpl = _rng_pick(_SLEEP_TEMPLATES, seed + 1)
        return tmpl.format(
            start_clock=start_clock,
            end_clock=end_clock,
            hours=sleep_window,
            note=note,
        )

    def _transit_text(
        self, entry: ActivityEntry, location_map: Dict[str, Location]
    ) -> str:
        from_name = location_map.get(
            entry.metadata.get("from_location_id", ""), None
        )
        to_loc = location_map.get(entry.location_id)
        from_display = from_name.name if from_name else "home"
        to_display = to_loc.name if to_loc else "the destination"
        minutes = entry.metadata.get("transit_minutes", 15)
        seed = hash(entry.entry_id) % 1000
        tmpl = _rng_pick(_TRANSIT_TEMPLATES, seed)
        return tmpl.format(
            from_name=from_display,
            to_name=to_display,
            minutes=minutes,
        )

    def _errand_text(self, entry: ActivityEntry, loc: Optional[Location]) -> str:
        location_name = loc.name if loc else "the shop"
        seed = hash(entry.entry_id) % 1000
        tmpl = _rng_pick(_ERRAND_TEMPLATES, seed)
        return tmpl.format(location_name=location_name)

    def _dialogue_text(
        self,
        entry: ActivityEntry,
        persona: PersonaCard,
        loc: Optional[Location],
        time_range: Tuple[float, float],
    ) -> str:
        """Template-based dialogue memory; uses extracted_facts hint if available."""
        names = _format_participant_names(entry.participants, entry.metadata.get("participant_names", {}))
        location_name = loc.name if loc else "somewhere"
        time_label = _time_label(entry.start_time, time_range, persona)
        fact_hint = entry.metadata.get("fact_hint", "")
        seed = hash(entry.entry_id) % 1000
        tmpl = _rng_pick(_DIALOGUE_TEMPLATES, seed)
        return tmpl.format(
            names=names,
            location_name=location_name,
            time_label=time_label,
            fact_hint=fact_hint,
        ).strip()

    def _role_conflict_text(self, entry: ActivityEntry) -> str:
        """Template-based memory for role_conflict entries."""
        winner = entry.metadata.get("winner_group_name", "another commitment")
        loser = entry.metadata.get("loser_group_name", "a meeting")
        seed = hash(entry.entry_id) % 1000
        tmpl = _rng_pick(_ROLE_CONFLICT_TEMPLATES, seed)
        return tmpl.format(winner=winner, loser=loser)

    def _solo_template_fallback(
        self,
        entry: ActivityEntry,
        persona: PersonaCard,
        loc: Optional[Location],
        time_range: Tuple[float, float],
    ) -> str:
        """Dry-run fallback for solo/group_meeting: plausible but template-based."""
        loc_name = loc.name if loc else "home"
        loc_type = loc.location_type if loc else "home_private"
        time_label = _time_label(entry.start_time, time_range, persona)
        duration = _duration_str((entry.end_time - entry.start_time) / max(time_range[1] - time_range[0], 1) * 16)
        if loc_type == "home_private":
            return f"Spent {duration} at home {time_label}. Had some time to myself."
        if loc_type == "workplace":
            return f"Worked at {loc_name} {time_label} for {duration}."
        if entry.activity_type == "group_meeting":
            return f"Met with the group at {loc_name} {time_label}. {duration} of discussion."
        return f"Spent {duration} at {loc_name} {time_label}."

    # ------------------------------------------------------------------
    # LLM task builder
    # ------------------------------------------------------------------

    def _build_llm_task(
        self,
        entry: ActivityEntry,
        persona: PersonaCard,
        loc: Optional[Location],
        time_range: Tuple[float, float],
    ) -> Dict[str, Any]:
        traits = ", ".join(persona.personality_traits[:3]) if persona.personality_traits else "curious"
        vocab = _EDU_VOCAB.get(persona.education_level.lower(), "match vocabulary to their background")
        duration_hours = (entry.end_time - entry.start_time) / max(time_range[1] - time_range[0], 1) * 16
        sim_duration = time_range[1] - time_range[0]

        # Context: what was happening just before/after
        activity_label = {
            "solo": "free / solo time",
            "group_meeting": "group meeting / social gathering",
        }.get(entry.activity_type, entry.activity_type)

        context = entry.description or f"{activity_label} at {loc.name if loc else 'unknown location'}"

        user_prompt = MEMORY_USER_TEMPLATE.format(
            name=persona.name,
            age=persona.age,
            occupation=persona.occupation,
            education_level=persona.education_level or "unspecified",
            vocab_register=vocab,
            traits=traits,
            speaking_style=(persona.speaking_style or "")[:150],
            routine=(persona.daily_routine_notes or "")[:100],
            activity_type=activity_label,
            location_name=loc.name if loc else "unknown",
            location_desc=(loc.description or loc.location_type) if loc else "",
            time_label=_time_label(entry.start_time, time_range, persona),
            duration_str=_duration_str(duration_hours),
            context=context,
            first_name=_first_name(persona.name),
        )
        return {
            "system": MEMORY_SYSTEM_PROMPT,
            "user": user_prompt,
            "max_tokens": 8192,
            "tags": {"phase": "memory_fill"},
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _time_label(sim_time: float, time_range: Tuple[float, float], persona: PersonaCard) -> str:
    """Map a simulation timestamp to a human-readable time-of-day label."""
    sim_start, sim_end = time_range
    sim_duration = max(sim_end - sim_start, 1.0)
    fraction = (sim_time - sim_start) / sim_duration

    wake = persona.sleep_end_hour if persona.sleep_end_hour else 7.0
    sleep_start = persona.sleep_start_hour if persona.sleep_start_hour else 23.0
    active = (sleep_start - wake) % 24 or 16.0
    hour = wake + fraction * active

    if hour < 9:   return "early in the morning"
    if hour < 11:  return "mid-morning"
    if hour < 13:  return "around midday"
    if hour < 15:  return "early afternoon"
    if hour < 18:  return "in the afternoon"
    if hour < 21:  return "in the evening"
    return "late at night"


def _format_participant_names(
    participant_ids: List[str],
    name_map: Dict[str, str],
) -> str:
    """Format participant agent IDs as readable names."""
    names = []
    for pid in participant_ids[:3]:
        if pid in name_map:
            names.append(name_map[pid].split()[0])
        else:
            # Slug to display: "maya_chen" -> "Maya"
            names.append(pid.split("_")[0].capitalize())
    if not names:
        return "someone"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]
