"""Generate world events with timestamps, content, and visibility assignments.

Events drive the simulation by giving agents things to discuss.
Visibility is controlled by event category and social graph structure.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

from MASim.core.schema import EventCategory, INTEREST_DOMAINS, WorldEvent
from MASim.ground_truth.json_parser import parse_json_object
from MASim.prompts import (
    EVENT_CATEGORY_DESCRIPTIONS as _STRING_CATEGORY_DESCRIPTIONS,
    EVENT_DYADIC_USER as DYADIC_EVENT_USER_TEMPLATE,
    EVENT_PRIVATE_USER as PRIVATE_EVENT_USER_TEMPLATE,
    EVENT_SYSTEM as EVENT_SYSTEM_PROMPT,
    EVENT_USER as EVENT_USER_TEMPLATE,
)
from MASim.utils.logging import get_logger

if TYPE_CHECKING:
    from MASim.core.schema import EncounterWindow

log = get_logger(__name__)

# Map EventCategory enum → description (prompts.py stores string-keyed version)
CATEGORY_DESCRIPTIONS = {
    EventCategory[k.upper()]: v for k, v in _STRING_CATEGORY_DESCRIPTIONS.items()
}

# Default event category distribution
DEFAULT_CATEGORY_WEIGHTS = {
    EventCategory.GLOBAL: 0.15,
    EventCategory.COMMUNITY: 0.35,
    EventCategory.DYADIC: 0.35,
    EventCategory.PRIVATE: 0.15,
}


class EventFactory:
    """Generate world events for the simulation."""

    def __init__(self, llm_client: Any, seed: int = 42):
        self.llm_client = llm_client
        self._rng = np.random.default_rng(seed)

    def generate_events(
        self,
        n_events: int,
        social_graph: nx.Graph,
        time_range: tuple = (0.0, 1000.0),
        category_weights: Optional[Dict[EventCategory, float]] = None,
        dry_run: bool = False,
        agent_names: Optional[Dict[str, str]] = None,
        encounter_windows: Optional[List["EncounterWindow"]] = None,
    ) -> List[WorldEvent]:
        """Generate a schedule of world events.

        Args:
            n_events: Number of events to generate.
            social_graph: The agent social graph for visibility assignment.
            time_range: (start, end) timestamps for event scheduling.
            category_weights: Distribution over event categories.
            dry_run: If True, generate placeholder events.
            agent_names: Mapping of agent_id -> display name.  When provided,
                DYADIC and PRIVATE event prompts include the actual agent names
                so the generated content references real simulation characters.
            encounter_windows: Optional list of co-location windows from the
                skeleton schedule.  When provided, DYADIC and COMMUNITY event
                timestamps and agent assignments are grounded in real encounters.

        Returns:
            Sorted list of WorldEvent instances.
        """
        if category_weights is None:
            category_weights = DEFAULT_CATEGORY_WEIGHTS
        if agent_names is None:
            agent_names = {}

        agents = list(social_graph.nodes)
        categories = list(category_weights.keys())
        weights = np.array([category_weights[c] for c in categories])
        weights = weights / weights.sum()

        # Assign categories
        cat_indices = self._rng.choice(len(categories), size=n_events, p=weights)
        event_categories = [categories[i] for i in cat_indices]

        # Assign timestamps — stratified: one event per equal-width bin with
        # jitter, ensuring events are spread evenly across the time range.
        t_start, t_end = time_range
        bin_width = (t_end - t_start) / n_events
        timestamps = [
            t_start + i * bin_width + self._rng.uniform(0, bin_width)
            for i in range(n_events)
        ]

        # Pre-assign visibility for every event BEFORE building prompts so that
        # DYADIC/PRIVATE prompts can reference the actual assigned agents by name.
        visibilities: List[tuple] = []
        for i, cat in enumerate(event_categories):
            win = None
            if encounter_windows:
                win = _pick_grounded_window(encounter_windows, cat, self._rng)
            if win is not None:
                vis, meta = self._assign_visibility_from_window(cat, win, agents, social_graph)
                # Ground timestamp at midpoint of encounter window
                timestamps[i] = (win.start_time + win.end_time) / 2.0
            else:
                vis, meta = self._assign_visibility(cat, agents, social_graph)
            visibilities.append((vis, meta))

        # Sort by timestamp
        order = sorted(range(n_events), key=lambda k: timestamps[k])
        timestamps = [timestamps[k] for k in order]
        event_categories = [event_categories[k] for k in order]
        visibilities = [visibilities[k] for k in order]

        # Build LLM prompts for event content
        domain_list = ", ".join(INTEREST_DOMAINS)
        tasks = []
        for cat, (visibility, metadata) in zip(event_categories, visibilities):
            cat_desc = CATEGORY_DESCRIPTIONS[cat]

            if cat == EventCategory.DYADIC and agent_names:
                pair = metadata.get("target_pair", [])
                name_a = agent_names.get(pair[0], pair[0]) if len(pair) > 0 else "Person A"
                name_b = agent_names.get(pair[1], pair[1]) if len(pair) > 1 else "Person B"
                prompt = DYADIC_EVENT_USER_TEMPLATE.format(
                    agent_a=name_a, agent_b=name_b, category_desc=cat_desc,
                    interest_domains=domain_list,
                )
            elif cat == EventCategory.PRIVATE and agent_names:
                target = metadata.get("target_agent", agents[0] if agents else "")
                name = agent_names.get(target, target)
                prompt = PRIVATE_EVENT_USER_TEMPLATE.format(
                    agent_name=name, category_desc=cat_desc,
                    interest_domains=domain_list,
                )
            else:
                prompt = EVENT_USER_TEMPLATE.format(
                    category=cat.value, category_desc=cat_desc,
                    interest_domains=domain_list,
                )
            tasks.append({"system": EVENT_SYSTEM_PROMPT, "user": prompt, "tags": {"phase": "event_gen"}})

        if dry_run:
            contents = [
                {
                    "event_type": f"{cat.value}_event",
                    "content": f"[DRY RUN] A {cat.value} event occurred.",
                    "interest_domain": INTEREST_DOMAINS[i % len(INTEREST_DOMAINS)],
                }
                for i, cat in enumerate(event_categories)
            ]
        else:
            log.info("Generating %d events via LLM...", n_events)
            responses = self.llm_client.generate_batch(tasks)
            contents = [self._parse_event(r, i) for i, r in enumerate(responses)]

        # Build WorldEvent objects (visibility already pre-assigned above)
        events = []
        for i, (ts, cat, content, (visibility, metadata)) in enumerate(
            zip(timestamps, event_categories, contents, visibilities)
        ):
            # Validate interest_domain; fall back to daily_logistics if missing/invalid
            raw_domain = content.get("interest_domain", "")
            if raw_domain not in INTEREST_DOMAINS:
                raw_domain = "daily_logistics"
            event = WorldEvent(
                event_id=f"evt_{i:06d}",
                timestamp=float(ts),
                event_type=content.get("event_type", "unknown"),
                content=content.get("content", ""),
                visibility_mask=visibility,
                category=cat,
                interest_domain=raw_domain,
                metadata=metadata,
            )
            events.append(event)

        log.info(
            "Generated %d events: %s",
            len(events),
            {cat.value: sum(1 for e in events if e.category == cat) for cat in EventCategory},
        )
        return events

    def _parse_event(self, response: str, index: int) -> Dict[str, str]:
        """Parse LLM response into event content."""
        try:
            return parse_json_object(response)
        except (ValueError, TypeError):
            log.warning("Failed to parse event %d, using fallback", index)
            return {"event_type": "generic", "content": response[:200]}

    def _assign_visibility(
        self,
        category: EventCategory,
        agents: List[str],
        graph: nx.Graph,
    ) -> tuple:
        """Assign visibility mask based on category and graph.

        Returns (visibility_mask: Set[str], metadata: Dict).
        """
        metadata: Dict[str, Any] = {}

        if category == EventCategory.GLOBAL:
            return set(agents), metadata

        if category == EventCategory.PRIVATE:
            target = self._rng.choice(agents)
            metadata["target_agent"] = target
            return {target}, metadata

        if category == EventCategory.DYADIC:
            # Pick a random edge (connected pair)
            edges = list(graph.edges)
            if edges:
                u, v = edges[self._rng.integers(len(edges))]
            else:
                u, v = self._rng.choice(agents, size=2, replace=False)
            metadata["target_pair"] = [u, v]
            return {u, v}, metadata

        if category == EventCategory.COMMUNITY:
            # Pick a random agent and their ego-graph neighborhood
            source = self._rng.choice(agents)
            metadata["source_agent"] = source
            ego_g = nx.ego_graph(graph, source, radius=2)
            return set(ego_g.nodes), metadata

        return set(agents), metadata

    def _assign_visibility_from_window(
        self,
        category: EventCategory,
        window: "EncounterWindow",
        agents: List[str],
        graph: nx.Graph,
    ) -> tuple:
        """Assign visibility using a pre-selected encounter window.

        Returns (visibility_mask: Set[str], metadata: Dict).
        """
        metadata: Dict[str, Any] = {"grounded_window_id": window.window_id}

        if category == EventCategory.DYADIC:
            pair = window.agent_ids[:2]
            metadata["target_pair"] = pair
            return set(pair), metadata

        if category == EventCategory.COMMUNITY:
            source = window.agent_ids[0] if window.agent_ids else (agents[0] if agents else "")
            metadata["source_agent"] = source
            if source in graph:
                ego_g = nx.ego_graph(graph, source, radius=2)
                return set(ego_g.nodes), metadata
            return set(window.agent_ids), metadata

        # Fallback for other categories
        return self._assign_visibility(category, agents, graph)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _pick_grounded_window(
    windows: List["EncounterWindow"],
    category: EventCategory,
    rng: np.random.Generator,
) -> Optional["EncounterWindow"]:
    """Pick an encounter window suitable for the given event category.

    For DYADIC: any co_located window with exactly 2 agents.
    For COMMUNITY: any co_located window with 3+ agents.
    Returns None if no suitable window is found.
    """
    if category == EventCategory.DYADIC:
        candidates = [w for w in windows if w.encounter_type == "co_located" and len(w.agent_ids) >= 2]
    elif category == EventCategory.COMMUNITY:
        candidates = [w for w in windows if w.encounter_type == "co_located" and len(w.agent_ids) >= 3]
    else:
        return None

    if not candidates:
        return None
    idx = rng.integers(len(candidates))
    return candidates[idx]
