"""SocialKnowledgeInjector: seed each agent's knowledge with tiered public profiles.

Every agent has a `public_information` dict (built by PersonaFactory) with five
disclosure tiers.  At the start of the simulation — before any dialogue — each
agent is seeded with what they would plausibly already know about every other
agent, based on how close they are in the social graph:

  Graph edge weight → disclosure tier:
    >= 0.75  →  intimate     (full profile)
    >= 0.50  →  close        (most things)
    >= 0.25  →  active       (basics)
     > 0.0   →  acquaintance (name + job)
    no edge  →  stranger     (name only)

Facts are stored at timestamp=-1.0 so they pre-date all simulation events.
"""

from __future__ import annotations

from typing import Any, Dict, List

import networkx as nx

from MASim.core.agent import EntityAgent
from MASim.core.knowledge_state import KnownFact, KnowledgeState
from MASim.core.schema import PersonaCard
from MASim.utils.logging import get_logger

log = get_logger(__name__)

# Tier ordering from most to least disclosure
_TIERS = ("intimate", "close", "active", "acquaintance", "stranger")


def _weight_to_tier(weight: float) -> str:
    """Map a social graph edge weight to a disclosure tier name."""
    if weight >= 0.75:
        return "intimate"
    if weight >= 0.50:
        return "close"
    if weight >= 0.25:
        return "active"
    return "acquaintance"


def _render_public_info(name: str, info: Dict[str, Any]) -> List[str]:
    """Convert a public_information tier dict to natural-language KnownFact strings."""
    facts: List[str] = []

    if not info:
        return facts

    if "name" in info:
        facts.append(f"{name} is someone in your social world.")

    if "age" in info:
        facts.append(f"{name} is {info['age']} years old.")

    if "occupation" in info and info["occupation"]:
        facts.append(f"{name} works as a {info['occupation']}.")

    if "location" in info and info["location"]:
        facts.append(f"{name} is based in {info['location']}.")

    if "communication_style" in info and info["communication_style"]:
        facts.append(
            f"{name} tends to communicate in a {info['communication_style']} style."
        )

    if "personality" in info and info["personality"]:
        facts.append(f"{name} is often described as: {info['personality']}.")

    if "hobbies" in info and info["hobbies"]:
        hobbies = ", ".join(info["hobbies"][:3])
        facts.append(f"{name}'s interests include {hobbies}.")

    if "values" in info and info["values"]:
        vals = ", ".join(info["values"][:3])
        facts.append(f"{name} cares about: {vals}.")

    if "current_concerns" in info and info["current_concerns"]:
        concern = info["current_concerns"][0]
        facts.append(f"{name} has been dealing with: {concern}.")

    if "backstory" in info and info["backstory"]:
        facts.append(f"Background on {name}: {info['backstory'][:200]}")

    if "speaking_style" in info and info["speaking_style"]:
        facts.append(f"How {name} typically speaks: {info['speaking_style'][:150]}")

    if "sleep_schedule" in info and info["sleep_schedule"]:
        facts.append(f"{name}'s sleep schedule: {info['sleep_schedule']}.")

    if "work_schedule" in info and info["work_schedule"]:
        facts.append(f"{name} works: {info['work_schedule']}.")

    if "daily_routine" in info and info["daily_routine"]:
        facts.append(f"{name}'s daily routine: {info['daily_routine']}")

    if "relationships" in info and info["relationships"]:
        for rname, rdesc in list(info["relationships"].items())[:2]:
            facts.append(f"{name} has a relationship with {rname}: {rdesc}.")

    return facts


def _inject_for_pair(
    agent: EntityAgent,
    agent_id: str,
    neighbour_id: str,
    persona: PersonaCard,
    tier: str,
) -> int:
    """Inject one neighbour's public profile into one agent's knowledge. Returns fact count."""
    pub = persona.public_information
    if not pub:
        return 0

    info = pub.get(tier, pub.get("stranger", {}))
    facts = _render_public_info(persona.name, info)

    for i, content in enumerate(facts):
        fact_id = f"{agent_id}_pub_{tier}_{neighbour_id}_{i:03d}"
        fact = KnownFact(
            fact_id=fact_id,
            content=content,
            source_agent="world",
            session_id="",
            timestamp=-1.0,
            confidence=0.95,
        )
        agent.knowledge.add_fact(fact)
        agent.knowledge.add_entity_fact(persona.name.split()[0], fact_id)
        agent.knowledge.add_entity_fact(neighbour_id, fact_id)

    return len(facts)


def inject_social_knowledge(
    agents: Dict[str, EntityAgent],
    graph: nx.Graph,
    persona_map: Dict[str, PersonaCard],
) -> None:
    """Inject tiered public-profile KnownFacts into every agent's knowledge state.

    For each agent A:
      - Graph neighbours get knowledge at the tier matching their edge weight.
      - Non-neighbours (strangers) get the "stranger" tier (name only).

    Mutates agent knowledge states in-place.
    """
    total_injected = 0
    tier_counts: Dict[str, int] = {t: 0 for t in _TIERS}

    for agent_id, agent in agents.items():
        # --- Graph neighbours ---
        for neighbour_id in graph.neighbors(agent_id):
            if neighbour_id not in persona_map or neighbour_id == agent_id:
                continue
            weight = float(graph[agent_id][neighbour_id].get("weight", 1.0))
            tier = _weight_to_tier(weight)
            n = _inject_for_pair(agent, agent_id, neighbour_id,
                                 persona_map[neighbour_id], tier)
            total_injected += n
            tier_counts[tier] += n

        # --- Strangers (no graph edge) ---
        for other_id, persona in persona_map.items():
            if other_id == agent_id or graph.has_edge(agent_id, other_id):
                continue
            n = _inject_for_pair(agent, agent_id, other_id, persona, "stranger")
            total_injected += n
            tier_counts["stranger"] += n

    log.info(
        "SocialKnowledgeInjector: %d facts across %d agents | tiers: %s",
        total_injected,
        len(agents),
        {k: v for k, v in tier_counts.items() if v},
    )
