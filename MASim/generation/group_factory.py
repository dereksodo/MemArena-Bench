"""GroupFactory: deterministically infer groups and role assignments from personas + locations.

Inferred group types (Phase 1 — no LLM required):
  work_team    — agents who share an occupation domain (same workplace)
  hobby        — agents sharing a hobby keyword cluster
  neighbourhood — all agents in the same city (the broadest social layer)

Each group is assigned a home_location (from the location list) and a
simple meeting_schedule. Every group membership yields a PersonRole.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from MASim.core.schema import Group, Location, PersonaCard, PersonRole
from MASim.generation.location_factory import _occupation_domain
from MASim.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Hobby-cluster keywords
# ---------------------------------------------------------------------------

_HOBBY_CLUSTERS: Dict[str, List[str]] = {
    "music":    ["music", "guitar", "piano", "jazz", "band", "choir", "sing", "instrument"],
    "sport":    ["running", "cycling", "football", "basketball", "swimming", "yoga", "gym",
                 "climbing", "tennis", "hiking", "martial arts", "volleyball"],
    "creative": ["painting", "drawing", "photography", "pottery", "sculpture", "craft",
                 "knitting", "sewing", "woodwork", "design"],
    "outdoors": ["hiking", "camping", "gardening", "birdwatching", "fishing", "surfing",
                 "trail", "foraging"],
    "literary": ["reading", "writing", "poetry", "book club", "zine", "journalism",
                 "blogging", "fiction"],
    "gaming":   ["gaming", "board games", "tabletop", "chess", "puzzles", "esports", "rpg"],
    "food":     ["cooking", "baking", "food", "wine", "coffee", "cocktails", "restaurant"],
    "community":["volunteering", "community", "activism", "organising", "politics", "fundraising"],
}


def _hobby_cluster(hobbies: List[str]) -> List[str]:
    """Return a list of hobby-cluster labels that match the agent's hobbies."""
    text = " ".join(hobbies).lower()
    return [
        cluster
        for cluster, keywords in _HOBBY_CLUSTERS.items()
        if any(kw in text for kw in keywords)
    ]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ---------------------------------------------------------------------------
# Role assignment helpers
# ---------------------------------------------------------------------------

_WORK_ROLES: Dict[str, List[str]] = {
    "tech":        ["senior developer", "developer", "junior developer", "analyst"],
    "healthcare":  ["charge nurse", "nurse", "junior nurse", "care assistant"],
    "education":   ["head teacher", "teacher", "teaching assistant", "student"],
    "creative":    ["lead artist", "artist", "apprentice"],
    "legal":       ["senior counsel", "associate", "paralegal"],
    "civil":       ["manager", "officer", "administrator", "intern"],
    "trades":      ["master tradesperson", "tradesperson", "apprentice"],
    "hospitality": ["head chef", "sous chef", "line cook", "server"],
    "sports":      ["coach", "athlete", "assistant coach"],
    "logistics":   ["logistics manager", "coordinator", "driver"],
    "general":     ["senior member", "member", "junior member"],
}

_HOBBY_ROLES: Dict[str, List[str]] = {
    "music":    ["lead musician", "musician", "vocalist", "rhythm section"],
    "sport":    ["captain", "player", "substitute"],
    "creative": ["lead artist", "artist", "apprentice"],
    "outdoors": ["trip leader", "member"],
    "literary": ["organiser", "member"],
    "gaming":   ["game master", "player"],
    "food":     ["host", "regular"],
    "community":["organiser", "coordinator", "volunteer", "member"],
}

_CONTEXT_REGISTER: Dict[str, str] = {
    "work_team":     "professional",
    "hobby":         "casual",
    "neighbourhood": "civil-casual",
    "family":        "intimate",
    "friendship_circle": "casual",
}


def _assign_work_role(domain: str, position: int) -> str:
    roles = _WORK_ROLES.get(domain, _WORK_ROLES["general"])
    return roles[min(position, len(roles) - 1)]


def _assign_hobby_role(cluster: str, position: int) -> str:
    roles = _HOBBY_ROLES.get(cluster, ["organiser", "member"])
    return roles[min(position, len(roles) - 1)]


def _find_location(
    locations: List[Location],
    *,
    location_type: Optional[str] = None,
    owner_agent_id: Optional[str] = None,
    affordance: Optional[str] = None,
) -> Optional[Location]:
    for loc in locations:
        if location_type and loc.location_type != location_type:
            continue
        if owner_agent_id and loc.owner_agent_id != owner_agent_id:
            continue
        if affordance and affordance not in loc.affordances:
            continue
        return loc
    return None


def _default_meeting_schedule(group_type: str, location_id: str) -> List[Dict]:
    schedules = {
        "work_team": [{
            "frequency": "weekday",
            "start_hour": 9.5,
            "duration_hours": 0.75,
            "location_id": location_id,
            "activity_description": "Morning stand-up / team check-in",
        }],
        "hobby": [{
            "frequency": "weekly",
            "day_of_week": 4,   # Friday
            "start_hour": 19.0,
            "duration_hours": 2.0,
            "location_id": location_id,
            "activity_description": "Regular hobby group meeting",
        }],
        "neighbourhood": [{
            "frequency": "monthly",
            "day_of_week": 6,   # Sunday
            "start_hour": 10.0,
            "duration_hours": 1.5,
            "location_id": location_id,
            "activity_description": "Neighbourhood association meeting",
        }],
    }
    return schedules.get(group_type, [])


# ---------------------------------------------------------------------------
# Main factory class
# ---------------------------------------------------------------------------

class GroupFactory:
    """Infer groups from personas and locations; assign PersonRoles."""

    def generate_groups(
        self,
        personas: List[PersonaCard],
        agent_ids: List[str],
        locations: List[Location],
    ) -> Tuple[List[Group], List[PersonRole]]:
        """Return (groups, all_person_roles)."""
        groups: List[Group] = []
        roles: List[PersonRole] = []

        # Build lookup helpers
        domain_to_agents: Dict[str, List[Tuple[str, PersonaCard]]] = {}
        for aid, p in zip(agent_ids, personas):
            domain = _occupation_domain(p.occupation)
            domain_to_agents.setdefault(domain, []).append((aid, p))

        cluster_to_agents: Dict[str, List[Tuple[str, PersonaCard]]] = {}
        for aid, p in zip(agent_ids, personas):
            for cluster in _hobby_cluster(p.hobbies):
                cluster_to_agents.setdefault(cluster, []).append((aid, p))

        # --- Work teams ---
        for domain, members in domain_to_agents.items():
            if len(members) < 2:
                continue   # lone wolves get no work-team group
            work_loc = _find_location(locations, location_type="workplace")
            loc_id = work_loc.location_id if work_loc else ""
            group_id = f"grp_work_{domain}"
            group = Group(
                group_id=group_id,
                name=f"{domain.replace('_', ' ').title()} Team",
                group_type="work_team",
                members={aid: _assign_work_role(domain, i) for i, (aid, _) in enumerate(members)},
                home_location_id=loc_id,
                meeting_schedule=_default_meeting_schedule("work_team", loc_id),
                shared_history=[
                    f"The {domain} team has been working together for over a year.",
                    "They share a group chat and occasionally get lunch together.",
                ],
                description=f"Colleagues in the {domain} domain who share a workplace.",
            )
            groups.append(group)
            if work_loc:
                work_loc.group_ids.append(group_id)

            for i, (aid, p) in enumerate(members):
                roles.append(PersonRole(
                    agent_id=aid,
                    group_id=group_id,
                    role_name=_assign_work_role(domain, i),
                    context_register=_CONTEXT_REGISTER["work_team"],
                    responsibilities=["attend stand-ups", "collaborate on projects"],
                    relationships_in_group={
                        other_aid: "colleague"
                        for j, (other_aid, _) in enumerate(members)
                        if other_aid != aid
                    },
                ))

        # --- Hobby groups ---
        for cluster, members in cluster_to_agents.items():
            if len(members) < 2:
                continue
            # Find a suitable third place
            event_loc = _find_location(locations, location_type="third_place_event")
            regular_loc = _find_location(locations, location_type="third_place_regular")
            hobby_loc = event_loc or regular_loc
            loc_id = hobby_loc.location_id if hobby_loc else ""
            group_id = f"grp_hobby_{cluster}"
            group = Group(
                group_id=group_id,
                name=f"{cluster.title()} Enthusiasts",
                group_type="hobby",
                members={aid: _assign_hobby_role(cluster, i) for i, (aid, _) in enumerate(members)},
                home_location_id=loc_id,
                meeting_schedule=_default_meeting_schedule("hobby", loc_id),
                shared_history=[
                    f"The group formed a couple of years ago through mutual interest in {cluster}.",
                    "They have an informal social media group and occasionally attend events together.",
                ],
                description=f"A group of agents who share a passion for {cluster}.",
            )
            groups.append(group)
            if hobby_loc:
                hobby_loc.group_ids.append(group_id)

            for i, (aid, p) in enumerate(members):
                roles.append(PersonRole(
                    agent_id=aid,
                    group_id=group_id,
                    role_name=_assign_hobby_role(cluster, i),
                    context_register=_CONTEXT_REGISTER["hobby"],
                    responsibilities=["attend regular meetups"],
                    relationships_in_group={
                        other_aid: "fellow enthusiast"
                        for other_aid, _ in members
                        if other_aid != aid
                    },
                ))

        # --- Neighbourhood group (all agents, if any share a city) ---
        city_to_agents: Dict[str, List[Tuple[str, PersonaCard]]] = {}
        for aid, p in zip(agent_ids, personas):
            city = p.demographics.get("location", "").split(",")[0].strip() or "Simville"
            city_to_agents.setdefault(city, []).append((aid, p))

        for city, members in city_to_agents.items():
            if len(members) < 2:
                continue
            outdoor_loc = _find_location(locations, location_type="outdoor")
            community_loc = _find_location(
                locations, location_type="third_place_event",
            )
            neigh_loc = community_loc or outdoor_loc
            loc_id = neigh_loc.location_id if neigh_loc else ""
            group_id = f"grp_neighbourhood_{_slug(city)}"
            group = Group(
                group_id=group_id,
                name=f"{city} Neighbourhood",
                group_type="neighbourhood",
                members={aid: "resident" for aid, _ in members},
                home_location_id=loc_id,
                meeting_schedule=_default_meeting_schedule("neighbourhood", loc_id),
                shared_history=[
                    f"Residents of the same neighbourhood in {city}.",
                    "Some know each other well; others are nodding acquaintances.",
                ],
                description=f"All agents living in {city}.",
            )
            groups.append(group)
            if neigh_loc:
                neigh_loc.group_ids.append(group_id)

            for aid, _ in members:
                roles.append(PersonRole(
                    agent_id=aid,
                    group_id=group_id,
                    role_name="resident",
                    context_register=_CONTEXT_REGISTER["neighbourhood"],
                    responsibilities=["participate in community events"],
                    relationships_in_group={
                        other_aid: "neighbour"
                        for other_aid, _ in members
                        if other_aid != aid
                    },
                ))

        log.info(
            "GroupFactory: %d groups, %d roles (%s)",
            len(groups),
            len(roles),
            ", ".join(f"{g.group_type}:{g.group_id}" for g in groups),
        )
        return groups, roles
