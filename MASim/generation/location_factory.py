"""LocationFactory: deterministic location graph with coordinate-based transit and cache.

Generates fictional but geographically coherent locations for the simulated city
"Millhaven", with neighborhoods, workplaces, and third places laid out on a
coordinate grid.  Transit times are computed from Euclidean distances with
tiered speed assumptions (walking / mixed / driving).

Results are cached under ``MASim/data/location_cache/`` so identical
(n_agents, seed) configurations skip regeneration entirely.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from MASim.core.schema import Location, PersonaCard
from MASim.utils.logging import get_logger

log = get_logger(__name__)

_CACHE_VERSION = 1
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "location_cache"

# ---------------------------------------------------------------------------
# Occupation-domain clustering
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "tech":        ["developer", "engineer", "data", "software", "analyst", "tech", "programmer"],
    "healthcare":  ["nurse", "doctor", "paramedic", "medical", "hospice", "therapist", "physician"],
    "education":   ["teacher", "professor", "lecturer", "academic", "tutor", "librarian", "student"],
    "creative":    ["musician", "artist", "designer", "architect", "translator", "writer", "chef"],
    "legal":       ["lawyer", "attorney", "defender", "judge", "paralegal"],
    "civil":       ["civil servant", "government", "organiser", "social worker", "officer"],
    "trades":      ["plumber", "electrician", "contractor", "mechanic", "builder"],
    "hospitality": ["chef", "restaurant", "bar", "café", "barista", "sous chef"],
    "sports":      ["athlete", "coach", "trainer", "fitness"],
    "logistics":   ["logistics", "military", "driver", "supply"],
}


def _occupation_domain(occupation: str) -> str:
    """Map an occupation string to a broad domain label."""
    occ_lower = occupation.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in occ_lower for kw in keywords):
            return domain
    return "general"


# ---------------------------------------------------------------------------
# Fictional city data — "Millhaven"
# ---------------------------------------------------------------------------
# Coordinates are in a local km grid (origin = city centre).

CITY_NAME = "Millhaven"

# 30 neighborhoods spread across ~15 km radius
_NEIGHBORHOODS: List[Tuple[str, float, float, str]] = [
    # (name, x_km, y_km, street_template)
    ("Oldtown",           0.0,   0.0,  "{n} Cobblestone Lane"),
    ("Harbour Row",       1.2,  -0.8,  "{n} Harbour Road"),
    ("Bridgewater",      -1.0,   0.5,  "{n} Bridge Street"),
    ("Millbrook",         0.8,   1.5,  "{n} Mill Road"),
    ("Eastgate",          2.5,   0.3,  "{n} East Gate Terrace"),
    ("Westmere",         -2.2,  -0.4,  "{n} Westmere Avenue"),
    ("Clover Hill",       0.3,   2.8,  "{n} Clover Hill Drive"),
    ("Redstone",         -1.8,   2.0,  "{n} Redstone Way"),
    ("Ashford",           3.0,   1.8,  "{n} Ashford Crescent"),
    ("Thornfield",       -0.5,  -2.5,  "{n} Thornfield Road"),
    ("Riverside",         1.5,  -2.0,  "{n} River Walk"),
    ("Coppergate",       -3.0,   0.0,  "{n} Coppergate Close"),
    ("Kingswood",         2.0,   3.5,  "{n} Kingswood Lane"),
    ("Lakeside",         -2.5,   3.2,  "{n} Lakeside Parade"),
    ("Quarry End",        4.0,  -0.5,  "{n} Quarry Lane"),
    ("Fenway",           -4.0,   1.5,  "{n} Fen Road"),
    ("North Cross",       0.0,   4.5,  "{n} North Cross Street"),
    ("Glendale",          3.5,   3.0,  "{n} Glendale Avenue"),
    ("Sunnyside",        -1.5,  -3.5,  "{n} Sunnyside Terrace"),
    ("The Pines",         5.0,   1.0,  "{n} Pine Street"),
    ("Willow Green",     -3.5,  -2.0,  "{n} Willow Close"),
    ("Chapel Row",        1.0,  -3.8,  "{n} Chapel Row"),
    ("Ironworks",         4.5,  -2.5,  "{n} Foundry Road"),
    ("Blackthorn",       -5.0,   0.5,  "{n} Blackthorn Lane"),
    ("Moorgate",          0.5,   6.0,  "{n} Moorgate Rise"),
    ("Silver Creek",      5.5,   3.5,  "{n} Silver Creek Road"),
    ("Heath End",        -4.5,  -3.0,  "{n} Heath End Way"),
    ("Stonebridge",       2.5,  -4.5,  "{n} Stonebridge Walk"),
    ("Fox Hollow",       -2.0,   5.0,  "{n} Fox Hollow Road"),
    ("Cedar Park",        6.0,   0.0,  "{n} Cedar Avenue"),
]

# Workplace templates by domain (name, x, y, address, description)
_WORKPLACE_TEMPLATES: Dict[str, Tuple[str, float, float, str, str]] = {
    "tech":        ("Millhaven Tech Hub",          0.5,   0.5,  "12 Innovation Row, Oldtown",
                    "A refurbished warehouse converted into open-plan offices and server rooms."),
    "healthcare":  ("Millhaven General Hospital",  -1.0,   1.2,  "1 Hospital Drive, Bridgewater",
                    "The city's main hospital — a sprawling complex with an A&E wing and rooftop helipad."),
    "education":   ("Millhaven Academy",            1.0,   2.0,  "5 Scholars Row, Millbrook",
                    "A mixed secondary school with a well-stocked library and creaky auditorium."),
    "creative":    ("The Forge Arts Collective",   -0.8,  -0.3,  "8 Gallery Lane, Bridgewater",
                    "A cooperative studio space with kilns, a darkroom, and paint-splattered floors."),
    "legal":       ("Millhaven Legal Centre",       0.2,  -0.5,  "3 Justice Square, Oldtown",
                    "A limestone office building shared by solicitors, barristers, and a notary."),
    "civil":       ("Millhaven Council Offices",    0.0,   0.8,  "1 Civic Plaza, Oldtown",
                    "The municipal government building — brutalist concrete and fluorescent lighting."),
    "trades":      ("Millhaven Trades Workshop",    3.5,  -1.0,  "17 Yard Road, Quarry End",
                    "An industrial estate with workshops, tool stores, and a shared loading bay."),
    "hospitality": ("The Millhaven Kitchen",       -0.5,  -1.0,  "22 Market Lane, Thornfield",
                    "A busy commercial kitchen that supplies three restaurant fronts on the same block."),
    "sports":      ("Millhaven Sports Complex",     2.0,   0.0,  "1 Stadium Road, Eastgate",
                    "A multi-sport facility with an indoor pool, running track, and five-a-side pitches."),
    "logistics":   ("Millhaven Logistics Depot",    5.0,  -2.0,  "1 Freight Yard, Ironworks",
                    "A distribution hub with loading docks, cold storage, and a small dispatch office."),
    "general":     ("Millhaven Office Building",    0.3,   0.3,  "10 Commerce Street, Oldtown",
                    "A generic multi-tenant office block with shared meeting rooms and a ground-floor deli."),
}

# Third places — fixed venues
_THIRD_PLACES: List[Dict[str, Any]] = [
    {
        "name": "Groundwork Coffee",         "x": -0.3, "y": -0.2,
        "location_type": "third_place_regular", "privacy_level": "public",
        "capacity": 40, "opening_hours": [7, 22],
        "affordances": ["socialise", "eat", "work"],
        "address": "4 Market Lane, Oldtown",
        "description": "A popular independent café with mismatched furniture and reliable Wi-Fi.",
    },
    {
        "name": "Millhaven Community Centre",  "x": 0.5, "y": 1.0,
        "location_type": "third_place_event", "privacy_level": "public",
        "capacity": 120, "opening_hours": [8, 23],
        "affordances": ["socialise", "leisure", "events"],
        "address": "2 Civic Plaza, Oldtown",
        "description": "A multi-purpose hall used for classes, rehearsals, and neighbourhood meetings.",
    },
    {
        "name": "The Anchor & Crown",          "x": 1.0, "y": -1.0,
        "location_type": "third_place_regular", "privacy_level": "public",
        "capacity": 60, "opening_hours": [12, 24],
        "affordances": ["socialise", "eat"],
        "address": "15 Harbour Road, Harbour Row",
        "description": "A neighbourhood pub — loud on Fridays, quiet enough for conversation on weekdays.",
    },
    {
        "name": "Heron Park",                  "x": -1.5, "y": 1.0,
        "location_type": "outdoor", "privacy_level": "public",
        "capacity": 500, "opening_hours": [0, 24],
        "affordances": ["leisure", "transit", "socialise"],
        "address": "Heron Park, Bridgewater",
        "description": "A well-maintained urban park with a bandstand and a popular Saturday market.",
    },
    {
        "name": "Millhaven Public Library",    "x": 0.0, "y": 0.5,
        "location_type": "third_place_regular", "privacy_level": "public",
        "capacity": 80, "opening_hours": [9, 21],
        "affordances": ["work", "socialise", "leisure"],
        "address": "7 Scholars Row, Oldtown",
        "description": "A Carnegie-era library with high ceilings, study nooks, and free community Wi-Fi.",
    },
    {
        "name": "Iron Rail Gym",               "x": 2.5, "y": -0.8,
        "location_type": "third_place_regular", "privacy_level": "semi-public",
        "capacity": 45, "opening_hours": [6, 22],
        "affordances": ["leisure", "socialise"],
        "address": "9 East Gate Terrace, Eastgate",
        "description": "A no-frills gym with free weights, cardio machines, and a boxing ring in the back.",
    },
    {
        "name": "The Copper Kettle",           "x": -2.0, "y": -0.5,
        "location_type": "third_place_regular", "privacy_level": "public",
        "capacity": 35, "opening_hours": [8, 20],
        "affordances": ["socialise", "eat"],
        "address": "3 Coppergate Close, Coppergate",
        "description": "A tea room with scones, board games on the shelf, and a cat that lives under the counter.",
    },
    {
        "name": "Starlight Cinema",            "x": 0.8, "y": -2.0,
        "location_type": "third_place_event", "privacy_level": "public",
        "capacity": 200, "opening_hours": [14, 24],
        "affordances": ["leisure", "socialise"],
        "address": "11 River Walk, Riverside",
        "description": "A two-screen independent cinema showing first-run and classic double bills.",
    },
]


# ---------------------------------------------------------------------------
# Coordinate-based transit
# ---------------------------------------------------------------------------

def _euclidean_km(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in km on the local grid."""
    return math.hypot(x2 - x1, y2 - y1)


def _distance_to_minutes(km: float) -> int:
    """Convert km distance to travel minutes with tiered speed.

    <1 km  → walking at 5 km/h  → 12 min/km
    1-5 km → mixed at 10 km/h   → 6 min/km
    >5 km  → driving at 30 km/h → 2 min/km
    Clamped to [3, 60].
    """
    if km < 1.0:
        minutes = km * 12.0
    elif km < 5.0:
        minutes = 12.0 + (km - 1.0) * 6.0
    else:
        minutes = 12.0 + 24.0 + (km - 5.0) * 2.0
    return max(3, min(60, int(round(minutes))))


def _build_coord_transit(
    locations: List[Location],
    coords: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, int]]:
    """Full transit matrix from coordinate map."""
    ids = [loc.location_id for loc in locations]
    matrix: Dict[str, Dict[str, int]] = {}
    for a in ids:
        matrix[a] = {}
        for b in ids:
            if a == b:
                matrix[a][b] = 0
                continue
            ca, cb = coords.get(a), coords.get(b)
            if ca and cb:
                km = _euclidean_km(ca[0], ca[1], cb[0], cb[1])
                matrix[a][b] = _distance_to_minutes(km)
            else:
                matrix[a][b] = 15  # fallback
    return matrix


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(n_agents: int, seed: int) -> str:
    return f"n{n_agents}_s{seed}.json"


def _load_cache(n_agents: int, seed: int) -> dict | None:
    """Return cached dict or None if miss / version mismatch."""
    path = _CACHE_DIR / _cache_key(n_agents, seed)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("version") != _CACHE_VERSION:
            log.info("Cache version mismatch, regenerating.")
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cache read failed (%s), regenerating.", exc)
        return None


def _write_cache(n_agents: int, seed: int, payload: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / _cache_key(n_agents, seed)
    payload["version"] = _CACHE_VERSION
    path.write_text(json.dumps(payload, indent=1))
    log.info("Wrote location cache to %s", path)


# ---------------------------------------------------------------------------
# LocationFactory
# ---------------------------------------------------------------------------

class LocationFactory:
    """Generate Location objects and a transit matrix from agent personas.

    All generation is now deterministic (no LLM calls). Results are cached
    so identical (n_agents, seed) configurations are instant on re-run.
    """

    def __init__(self, llm_client: Any = None, seed: int = 42):
        self.seed = seed

    def generate_locations(
        self,
        personas: List[PersonaCard],
        agent_ids: List[str],
        dry_run: bool = False,
    ) -> Tuple[List[Location], Dict[str, Dict[str, int]]]:
        """Return (locations, transit_matrix).

        transit_matrix[loc_id_a][loc_id_b] = travel time in minutes.
        Both ``dry_run=True`` and ``dry_run=False`` now follow the same
        deterministic path (no LLM calls).
        """
        n_agents = len(personas)
        cached = _load_cache(n_agents, self.seed)
        if cached is not None:
            log.info("Loaded %d locations from cache (n=%d, seed=%d).",
                     len(cached["locations"]), n_agents, self.seed)
            return self._restore_from_cache(cached, personas, agent_ids)

        locations, transit_matrix, coords = self._build_locations(personas, agent_ids)

        # Write cache
        _write_cache(n_agents, self.seed, {
            "locations": [loc.to_dict() for loc in locations],
            "transit_matrix": transit_matrix,
            "coords": {k: list(v) for k, v in coords.items()},
            "agent_home_map": {aid: p.home_location_id
                               for aid, p in zip(agent_ids, personas)},
        })

        log.info("Generated %d locations for %d agents.", len(locations), n_agents)
        return locations, transit_matrix

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _build_locations(
        self,
        personas: List[PersonaCard],
        agent_ids: List[str],
    ) -> Tuple[List[Location], Dict[str, Dict[str, int]], Dict[str, Tuple[float, float]]]:
        """Build all locations deterministically from fictional city data."""
        rng = random.Random(self.seed)
        locations: List[Location] = []
        coords: Dict[str, Tuple[float, float]] = {}

        # --- Homes (one per agent) ---
        neighborhoods = list(_NEIGHBORHOODS)
        rng.shuffle(neighborhoods)

        for i, (persona, aid) in enumerate(zip(personas, agent_ids)):
            nb_name, nb_x, nb_y, street_tmpl = neighborhoods[i % len(neighborhoods)]
            # Jitter within neighborhood (±0.3 km)
            jx = rng.uniform(-0.3, 0.3)
            jy = rng.uniform(-0.3, 0.3)
            hx, hy = nb_x + jx, nb_y + jy

            first_name = persona.name.split()[0] if persona.name else aid
            house_num = 10 + rng.randint(0, 90)
            loc_id = f"loc_home_{aid}"

            loc = Location(
                location_id=loc_id,
                name=f"{first_name}'s Home",
                location_type="home_private",
                address=street_tmpl.format(n=house_num) + f", {nb_name}",
                privacy_level="private",
                capacity=4,
                opening_hours=[0, 24],
                affordances=["sleep", "domestic", "socialise"],
                owner_agent_id=aid,
                description=f"{first_name}'s private residence in {nb_name}.",
            )
            locations.append(loc)
            coords[loc_id] = (hx, hy)
            persona.home_location_id = loc_id

        # --- Workplaces (one per occupied domain) ---
        domain_to_agents: Dict[str, List[str]] = {}
        for persona, aid in zip(personas, agent_ids):
            domain = _occupation_domain(persona.occupation)
            domain_to_agents.setdefault(domain, []).append(aid)

        for domain in sorted(domain_to_agents):
            tmpl = _WORKPLACE_TEMPLATES.get(domain, _WORKPLACE_TEMPLATES["general"])
            wp_name, wx, wy, wp_addr, wp_desc = tmpl
            loc_id = f"loc_work_{domain}"
            loc = Location(
                location_id=loc_id,
                name=wp_name,
                location_type="workplace",
                address=wp_addr,
                privacy_level="semi-public",
                capacity=50,
                opening_hours=[8, 19],
                affordances=["work", "professional conversation"],
                owner_agent_id="",
                description=wp_desc,
            )
            locations.append(loc)
            coords[loc_id] = (wx, wy)

        # --- Third places ---
        for tp in _THIRD_PLACES:
            slug = re.sub(r'[^a-z0-9]+', '_', tp["name"].lower()).strip('_')
            loc_id = f"loc_{slug}"
            loc = Location(
                location_id=loc_id,
                name=tp["name"],
                location_type=tp["location_type"],
                address=tp["address"],
                privacy_level=tp["privacy_level"],
                capacity=tp["capacity"],
                opening_hours=list(tp["opening_hours"]),
                affordances=list(tp["affordances"]),
                owner_agent_id="",
                description=tp["description"],
            )
            locations.append(loc)
            coords[loc_id] = (tp["x"], tp["y"])

        transit_matrix = _build_coord_transit(locations, coords)
        return locations, transit_matrix, coords

    # ------------------------------------------------------------------
    # Cache restoration
    # ------------------------------------------------------------------

    def _restore_from_cache(
        self,
        cached: dict,
        personas: List[PersonaCard],
        agent_ids: List[str],
    ) -> Tuple[List[Location], Dict[str, Dict[str, int]]]:
        """Rebuild Location objects from cached dicts and re-link home IDs."""
        locations = []
        for raw in cached["locations"]:
            loc = Location(
                location_id=raw["location_id"],
                name=raw["name"],
                location_type=raw["location_type"],
                address=raw.get("address", ""),
                privacy_level=raw.get("privacy_level", "public"),
                capacity=int(raw.get("capacity", 50)),
                opening_hours=list(raw.get("opening_hours", [0, 24])),
                affordances=list(raw.get("affordances", [])),
                owner_agent_id=raw.get("owner_agent_id", ""),
                description=raw.get("description", ""),
            )
            locations.append(loc)

        # Re-assign home_location_id on personas
        home_map = cached.get("agent_home_map", {})
        for persona, aid in zip(personas, agent_ids):
            if aid in home_map:
                persona.home_location_id = home_map[aid]

        return locations, cached["transit_matrix"]
