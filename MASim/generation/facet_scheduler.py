"""FacetScheduler: generate routine facet-sessions based on social distance.

For each simulated day, for each agent pair (A, B) with a social edge:
  1. Compute session probability K from Dunbar layer.
  2. Roll dice — if no session today, skip.
  3. Determine domain breadth N from Dunbar layer.
  4. Select N domains: prioritise overlap in hobbies/expertise/values, fill
     remaining slots randomly.
  5. For each domain, pick a routine prompt from the bank.
  6. Emit a ConversationSpec per domain.

All emitted specs are independent — full parallelism within a day.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from MASim.core.agent import EntityAgent
from MASim.core.scheduler import ConversationSpec
from MASim.core.schema import (
    DUNBAR_DOMAIN_BREADTH,
    DUNBAR_SESSION_PROB,
    FACET_MODALITY_BY_LAYER,
    INTEREST_DOMAINS,
    WorldEvent,
)
from MASim.generation.routine_prompts import ROUTINE_PROMPTS
from MASim.utils.logging import get_logger

log = get_logger(__name__)


def _agent_domain_affinity(persona: Any) -> Set[str]:
    """Extract interest domains an agent has natural affinity for.

    Maps persona fields (hobbies, expertise, values, occupation) to the
    canonical 16 domains using simple keyword overlap.
    """
    keywords = set()
    for field_val in (
        getattr(persona, "hobbies", []),
        getattr(persona, "expertise", []),
        getattr(persona, "values", []),
    ):
        for item in (field_val if isinstance(field_val, list) else [field_val]):
            keywords.update(str(item).lower().split())

    occ = getattr(persona, "occupation", "").lower()
    keywords.update(occ.split())

    # Simple mapping: if any keyword overlaps with the domain_id or its synonyms
    _DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
        "family":          {"family", "parent", "child", "kid", "marriage", "spouse", "sibling"},
        "work":            {"work", "job", "career", "office", "business", "professional", "manager", "engineer", "developer"},
        "health":          {"health", "fitness", "gym", "yoga", "medical", "exercise", "wellness", "running"},
        "finance":         {"finance", "money", "invest", "budget", "tax", "accounting", "bank", "economics"},
        "hobbies":         {"hobby", "craft", "art", "music", "paint", "draw", "guitar", "piano", "garden", "photography"},
        "politics":        {"politics", "government", "policy", "vote", "election", "activism", "civic"},
        "food_cooking":    {"food", "cooking", "baking", "recipe", "chef", "cuisine", "restaurant", "nutrition"},
        "travel":          {"travel", "trip", "adventure", "explore", "tourism", "backpack", "hiking"},
        "technology":      {"technology", "tech", "software", "hardware", "computer", "programming", "coding", "ai", "data"},
        "relationships":   {"relationship", "friendship", "dating", "social", "community", "love"},
        "education":       {"education", "learning", "teaching", "school", "university", "study", "academic", "research"},
        "entertainment":   {"entertainment", "movie", "film", "tv", "game", "gaming", "music", "theatre", "comedy"},
        "sports":          {"sport", "football", "basketball", "tennis", "soccer", "swimming", "athletics", "team"},
        "spirituality":    {"spiritual", "meditation", "faith", "religion", "mindfulness", "philosophy", "prayer"},
        "housing":         {"housing", "home", "apartment", "renovation", "interior", "property", "real estate", "rent"},
        "daily_logistics": {"logistics", "commute", "errand", "schedule", "routine", "organise", "organize", "plan"},
    }

    affinities: Set[str] = set()
    for domain, domain_kws in _DOMAIN_KEYWORDS.items():
        if keywords & domain_kws:
            affinities.add(domain)

    return affinities


def _select_domains(
    affinity_a: Set[str],
    affinity_b: Set[str],
    n_domains: int,
    rng: np.random.Generator,
) -> List[str]:
    """Select n_domains from the 16, prioritising shared affinities."""
    overlap = sorted(affinity_a & affinity_b)
    rng.shuffle(overlap)

    selected = overlap[:n_domains]
    if len(selected) >= n_domains:
        return selected[:n_domains]

    # Fill from union minus overlap
    union_rest = sorted((affinity_a | affinity_b) - set(selected))
    rng.shuffle(union_rest)
    selected.extend(union_rest[:n_domains - len(selected)])
    if len(selected) >= n_domains:
        return selected[:n_domains]

    # Fill from remaining domains
    remaining = sorted(set(INTEREST_DOMAINS) - set(selected))
    rng.shuffle(remaining)
    selected.extend(remaining[:n_domains - len(selected)])

    return selected[:n_domains]


def _get_dunbar_layer(
    agent_a: str,
    agent_b: str,
    graph: nx.Graph,
    agents: Dict[str, EntityAgent],
) -> int:
    """Determine the Dunbar layer between two agents.

    Uses the edge weight if available, otherwise falls back to the
    persona's dunbar_layer field (defaulting to layer 4 = acquaintance).
    """
    if graph.has_edge(agent_a, agent_b):
        weight = graph[agent_a][agent_b].get("weight", 0.0)
        # Map edge weight to Dunbar layer: higher weight = closer
        if weight >= 0.8:
            return 1  # intimate
        elif weight >= 0.5:
            return 2  # close
        elif weight >= 0.2:
            return 3  # active
        else:
            return 4  # acquaintance
    return 4  # no edge → acquaintance


class FacetScheduler:
    """Generate routine facet-session ConversationSpecs for each simulated day."""

    def __init__(
        self,
        agents: Dict[str, EntityAgent],
        graph: nx.Graph,
        rng: np.random.Generator,
        session_prob_overrides: Optional[Dict[int, float]] = None,
        domain_breadth_overrides: Optional[Dict[int, int]] = None,
    ):
        self.agents = agents
        self.graph = graph
        self.rng = rng
        self.session_prob = dict(DUNBAR_SESSION_PROB)
        self.domain_breadth = dict(DUNBAR_DOMAIN_BREADTH)
        if session_prob_overrides:
            self.session_prob.update(session_prob_overrides)
        if domain_breadth_overrides:
            self.domain_breadth.update(domain_breadth_overrides)

        # Pre-compute per-agent domain affinities
        self._affinities: Dict[str, Set[str]] = {}
        for agent_id, agent in agents.items():
            self._affinities[agent_id] = _agent_domain_affinity(agent.persona)

        # Pre-compute per-domain prompt indices for round-robin selection
        self._prompt_counters: Dict[str, int] = {d: 0 for d in INTEREST_DOMAINS}

    def _pick_routine_prompt(self, domain: str) -> str:
        """Pick the next routine prompt for a domain (round-robin)."""
        prompts = ROUTINE_PROMPTS.get(domain, [])
        if not prompts:
            return f"You and {{other}} are having a conversation about {domain}."
        idx = self._prompt_counters.get(domain, 0) % len(prompts)
        self._prompt_counters[domain] = idx + 1
        return prompts[idx]

    def _roll_facet_modality(self, layer: int) -> str:
        """Sample modality based on Dunbar layer (stronger ties → richer media)."""
        weights = FACET_MODALITY_BY_LAYER.get(layer, FACET_MODALITY_BY_LAYER[4])
        modalities = list(weights.keys())
        probs = np.array([weights[m] for m in modalities], dtype=float)
        probs /= probs.sum()
        return self.rng.choice(modalities, p=probs)

    def build_routine_specs_for_day(
        self,
        day_start: float,
        day_length: float = 1.0,
    ) -> List[ConversationSpec]:
        """Generate routine facet-session specs for a single day.

        Returns a list of ConversationSpec objects, each tagged with
        interest_domain, session_type="routine", and a routine_prompt.
        Modality is layer-dependent (Haythornthwaite 2005).
        """
        conv_duration = day_length * 0.03  # ~45 min in a 24h day
        day_end = day_start + day_length
        active_start = day_start + day_length * 0.08
        active_end = day_start + day_length * 0.90
        active_span = active_end - active_start

        specs: List[ConversationSpec] = []
        edges = list(self.graph.edges)

        for agent_a, agent_b in edges:
            if agent_a not in self.agents or agent_b not in self.agents:
                continue

            layer = _get_dunbar_layer(agent_a, agent_b, self.graph, self.agents)
            prob = self.session_prob.get(layer, 0.05)

            if self.rng.random() > prob:
                continue

            n_domains = self.domain_breadth.get(layer, 1)
            domains = _select_domains(
                self._affinities.get(agent_a, set()),
                self._affinities.get(agent_b, set()),
                n_domains,
                self.rng,
            )

            for i, domain in enumerate(domains):
                t_offset = active_span * (i + 0.5) / max(len(domains), 1)
                start_time = active_start + t_offset
                start_time += self.rng.uniform(-conv_duration, conv_duration) * 0.5
                start_time = max(active_start, min(start_time, active_end - conv_duration))
                end_time = start_time + conv_duration

                prompt = self._pick_routine_prompt(domain)
                name_a = self.agents[agent_a].persona.name or agent_a
                name_b = self.agents[agent_b].persona.name or agent_b
                first_a = name_a.split()[0] if name_a else agent_a
                first_b = name_b.split()[0] if name_b else agent_b
                resolved_prompt = prompt.replace("{self}", first_a).replace("{other}", first_b)

                modality = self._roll_facet_modality(layer)

                # For remote modality, no physical location needed
                if modality in ("text_message", "voice_message"):
                    location = ""
                else:
                    loc_a = self.agents[agent_a].persona.home_location_id
                    loc_b = self.agents[agent_b].persona.home_location_id
                    location = self.rng.choice([loc_a, loc_b]) if (loc_a and loc_b) else (loc_a or loc_b or "")

                specs.append(ConversationSpec(
                    spec_id=f"spec_facet_{uuid.uuid4().hex[:8]}",
                    participants=sorted([agent_a, agent_b]),
                    location_id=location,
                    start_time=start_time,
                    end_time=end_time,
                    trigger="routine_facet",
                    modality=modality,
                    is_group=False,
                    interest_domain=domain,
                    session_type="routine",
                    routine_prompt=resolved_prompt,
                ))

        specs.sort(key=lambda s: s.start_time)
        return specs

    def build_routine_specs(
        self,
        time_range: Tuple[float, float],
        day_length: float = 1.0,
    ) -> List[ConversationSpec]:
        """Generate routine facet-session specs for all days in the time range.

        Delegates to build_routine_specs_for_day() per day.
        """
        t_start, t_end = time_range
        n_days = max(1, int(round((t_end - t_start) / day_length)))
        all_specs: List[ConversationSpec] = []

        for day in range(n_days):
            day_start = t_start + day * day_length
            day_specs = self.build_routine_specs_for_day(day_start, day_length)
            all_specs.extend(day_specs)

        log.info(
            "FacetScheduler: %d routine specs across %d days, %d edges",
            len(all_specs), n_days, len(self.graph.edges),
        )
        return all_specs
