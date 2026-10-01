"""WorldBroadcaster: event scheduling and visibility-based distribution.

Controls which agents see which events based on event category
and the social graph structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import networkx as nx

from MASim.core.agent import EntityAgent
from MASim.core.schema import EventCategory, WorldEvent
from MASim.utils.logging import get_logger

log = get_logger(__name__)


class WorldBroadcaster:
    """Distributes world events to agents based on visibility rules.

    Event categories control visibility:
    - GLOBAL: all agents
    - COMMUNITY: agents within a graph community/cluster
    - DYADIC: a specific pair of agents
    - PRIVATE: a single agent
    """

    def __init__(
        self,
        event_schedule: List[WorldEvent],
        agents: Dict[str, EntityAgent],
        social_graph: nx.Graph,
    ):
        self.event_schedule = sorted(event_schedule, key=lambda e: e.timestamp)
        self.agents = agents
        self.social_graph = social_graph
        self.current_idx = 0
        self.broadcast_log: List[Dict[str, Any]] = []

    def broadcast_event(self, event: WorldEvent) -> List[str]:
        """Distribute a single event to visible agents.

        Returns list of agent_ids that received the event.
        """
        if event.visibility_mask:
            # Explicit visibility mask takes precedence
            recipients = [aid for aid in event.visibility_mask if aid in self.agents]
        else:
            # Derive from event category
            recipients = self._derive_recipients(event)

        for aid in recipients:
            self.agents[aid].receive_event(event)

        self.broadcast_log.append({
            "event_id": event.event_id,
            "timestamp": event.timestamp,
            "category": event.category.value,
            "n_recipients": len(recipients),
            "recipients": recipients,
        })

        log.debug(
            "Broadcast event %s (%s) to %d agents",
            event.event_id, event.category.value, len(recipients),
        )
        return recipients

    def _derive_recipients(self, event: WorldEvent) -> List[str]:
        """Derive recipient list from event category and graph structure."""
        all_agents = list(self.agents.keys())

        if event.category == EventCategory.GLOBAL:
            return all_agents

        if event.category == EventCategory.PRIVATE:
            # Assign to a single agent from metadata or first agent
            target = event.metadata.get("target_agent")
            if target and target in self.agents:
                return [target]
            return []

        if event.category == EventCategory.DYADIC:
            # Assign to a pair from metadata
            pair = event.metadata.get("target_pair", [])
            return [a for a in pair if a in self.agents]

        if event.category == EventCategory.COMMUNITY:
            # Find community containing the source agent
            source = event.metadata.get("source_agent")
            if not source or source not in self.social_graph:
                return all_agents  # fallback to global

            # Use ego graph at radius 2 as community
            ego_g = nx.ego_graph(self.social_graph, source, radius=2)
            return [n for n in ego_g.nodes if n in self.agents]

        return all_agents

    def step(self, timestamp: float) -> List[WorldEvent]:
        """Advance simulation to a timestamp, broadcasting due events.

        Returns list of events broadcast in this step.
        """
        broadcast = []
        while self.current_idx < len(self.event_schedule):
            event = self.event_schedule[self.current_idx]
            if event.timestamp > timestamp:
                break
            self.broadcast_event(event)
            broadcast.append(event)
            self.current_idx += 1
        return broadcast

    def get_remaining_events(self) -> List[WorldEvent]:
        """Get events not yet broadcast."""
        return self.event_schedule[self.current_idx:]

    def is_done(self) -> bool:
        """Check if all events have been broadcast."""
        return self.current_idx >= len(self.event_schedule)
