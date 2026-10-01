"""All shared dataclasses for the MASim framework."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

from memarena.text_cleaning import clean_llm_text


# ---------------------------------------------------------------------------
# Assistant ID helpers
# ---------------------------------------------------------------------------

def assistant_id_for(agent_id: str) -> str:
    """Per-agent assistant ID, e.g. 'assistant_maya_chen'."""
    return f"assistant_{agent_id}"


def is_assistant(participant_id: str) -> bool:
    """Check if a participant ID belongs to an AI assistant."""
    return participant_id.startswith("assistant_") or participant_id == "__assistant__"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventCategory(enum.Enum):
    """Visibility scope of a world event."""
    GLOBAL = "global"          # visible to all agents
    COMMUNITY = "community"    # visible to a subgraph / cluster
    DYADIC = "dyadic"          # visible to a pair
    PRIVATE = "private"        # visible to a single agent


# The 16 canonical interest domains for faceted sub-agents.
INTEREST_DOMAINS: List[str] = [
    "family", "work", "health", "finance", "hobbies", "politics",
    "food_cooking", "travel", "technology", "relationships",
    "education", "entertainment", "sports", "spirituality",
    "housing", "daily_logistics",
]

# Speaking-style register per interest domain (injected into facet system prompt).
DOMAIN_REGISTER: Dict[str, str] = {
    "family":           "warm, casual, emotionally open",
    "work":             "professional, measured, uses industry jargon",
    "health":           "concerned, empathetic, sometimes anxious",
    "finance":          "cautious, analytical, number-oriented",
    "hobbies":          "enthusiastic, relaxed, uses hobby-specific terms",
    "politics":         "opinionated but diplomatic, cites news",
    "food_cooking":     "sensory language, nostalgic, shares recipes",
    "travel":           "adventurous, descriptive, references places",
    "technology":       "precise, curious, explains things step-by-step",
    "relationships":    "reflective, gossipy, emotionally nuanced",
    "education":        "structured, mentoring tone, references learning",
    "entertainment":    "playful, pop-culture references, recommends things",
    "sports":           "competitive, stats-aware, team-oriented",
    "spirituality":     "reflective, philosophical, gentle",
    "housing":          "practical, domestic details, neighbourhood chat",
    "daily_logistics":  "matter-of-fact, planning-oriented, brief",
}

# Session probability by Dunbar layer (intimate→acquaintance).
DUNBAR_SESSION_PROB: Dict[int, float] = {
    1: 0.95,   # intimate
    2: 0.70,   # close
    3: 0.30,   # active
    4: 0.05,   # acquaintance
}

# How many interest domains a pair discusses per day, by Dunbar layer.
DUNBAR_DOMAIN_BREADTH: Dict[int, int] = {
    1: 16,     # intimate — all domains
    2: 9,      # close
    3: 4,      # active
    4: 1,      # acquaintance
}

# Facet-routine modality distribution by Dunbar layer.
# Haythornthwaite (2005): stronger ties use richer media channels.
FACET_MODALITY_BY_LAYER: Dict[int, Dict[str, float]] = {
    1: {"face_to_face": 0.60, "voice_message": 0.20, "text_message": 0.20},
    2: {"face_to_face": 0.50, "voice_message": 0.20, "text_message": 0.30},
    3: {"face_to_face": 0.30, "voice_message": 0.15, "text_message": 0.55},
    4: {"face_to_face": 0.15, "voice_message": 0.10, "text_message": 0.75},
}

# Default PP session-type proportions per day (social-science-informed).
# Granovetter (1973), Feld (1981), Baym (2015).
DEFAULT_PP_DAY_PROPORTIONS: Dict[str, float] = {
    "facet_routine":   0.40,   # tie-maintenance topical conversation
    "event_triggered": 0.15,   # news/event-driven talk
    "encounter":       0.25,   # unplanned co-location interaction
    "remote":          0.20,   # digitally-mediated contact
}


class Dimension(enum.Enum):
    """The evaluated internal task dimensions."""
    D1_CONFLICT = "d1_conflict"
    D2_ANAPHORA = "d2_anaphora"
    D3_CONFABULATION = "d3_confabulation"
    D4_PERMISSION = "d4_permission"
    D5_CLOZE = "d5_cloze"
    D6_METADATA = "d6_metadata"
    D7_QA = "d7_qa"
    D8_TEMPORAL = "d8_temporal"
    D10_COUNTERFACTUAL = "d10_counterfactual"


# ---------------------------------------------------------------------------
# Agent / Persona
# ---------------------------------------------------------------------------

@dataclass
class PersonaCard:
    """Rich persona description for an agent."""
    name: str = ""
    age: int = 30
    occupation: str = ""
    demographics: Dict[str, str] = field(default_factory=dict)
    personality_traits: List[str] = field(default_factory=list)
    expertise: List[str] = field(default_factory=list)
    communication_style: str = "neutral"
    education_level: str = ""  # e.g. "high school diploma", "bachelor's degree", "PhD"
    dunbar_layer: int = 0  # 0=ego, 1=intimate(5), 2=close(15), 3=active(50), 4=acquaintance(150)

    # Scheduling fields
    sleep_start_hour: float = 23.0       # 0-24; hour of night when agent goes to sleep
    sleep_end_hour: float = 7.0          # hour of morning when agent wakes
    work_schedule: str = "9-5 weekdays"  # e.g. "shift worker 6am-2pm", "freelance", "retired"
    home_location_id: str = ""           # assigned by LocationFactory
    daily_routine_notes: str = ""        # habitual patterns, favourite spots, morning rituals

    # Rich fields for deeper characterisation
    backstory: str = ""          # 2-3 sentence personal history
    speaking_style: str = ""     # detailed speech mannerisms, e.g. "uses rhetorical questions,
                                 # tends to go on tangents about childhood, rarely uses slang"
    hobbies: List[str] = field(default_factory=list)
    current_concerns: List[str] = field(default_factory=list)  # what's on their mind lately
    values: List[str] = field(default_factory=list)            # core beliefs / what they care about
    relationships: Dict[str, str] = field(default_factory=dict)  # name -> relationship description
    tech_affinity: float = 0.5  # 0.0 (technophobe) to 1.0 (digital native)

    # Tiered public profile: what this agent reveals at each social distance.
    # Keys: "intimate" | "close" | "active" | "acquaintance" | "stranger"
    # Populated by persona_factory.build_public_information() after persona creation.
    public_information: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Render persona card as a detailed text prompt for LLM role-play."""
        traits = ", ".join(self.personality_traits) if self.personality_traits else "none specified"
        expertise = ", ".join(self.expertise) if self.expertise else "none specified"
        demo = "; ".join(f"{k}: {v}" for k, v in self.demographics.items()) if self.demographics else "N/A"
        hobbies = ", ".join(self.hobbies) if self.hobbies else "not specified"
        concerns = "; ".join(self.current_concerns) if self.current_concerns else "nothing in particular"
        values = ", ".join(self.values) if self.values else "not specified"

        lines = [
            f"Name: {self.name}",
            f"Age: {self.age}",
            f"Occupation: {self.occupation}",
            f"Demographics: {demo}",
            f"Personality: {traits}",
            f"Expertise: {expertise}",
            f"Hobbies: {hobbies}",
            f"Values: {values}",
            f"Currently thinking about: {concerns}",
            f"Communication style: {self.communication_style}",
        ]
        if self.education_level:
            lines.append(f"Education: {self.education_level}")
        if self.work_schedule:
            lines.append(f"Work schedule: {self.work_schedule}")
        if self.daily_routine_notes:
            lines.append(f"Daily routine: {self.daily_routine_notes}")
        if self.speaking_style:
            lines.append(f"Speaking mannerisms: {self.speaking_style}")
        if self.backstory:
            lines.append(f"Background: {self.backstory}")
        if self.relationships:
            rels = "; ".join(f"{k} ({v})" for k, v in list(self.relationships.items())[:5])
            lines.append(f"Key relationships: {rels}")
        # Tech affinity label
        if self.tech_affinity >= 0.7:
            lines.append("Technology comfort: very comfortable with technology")
        elif self.tech_affinity >= 0.4:
            lines.append("Technology comfort: somewhat comfortable with technology")
        else:
            lines.append("Technology comfort: not very tech-savvy")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "age": self.age,
            "occupation": self.occupation,
            "demographics": self.demographics,
            "personality_traits": self.personality_traits,
            "expertise": self.expertise,
            "communication_style": self.communication_style,
            "education_level": self.education_level,
            "dunbar_layer": self.dunbar_layer,
            "sleep_start_hour": self.sleep_start_hour,
            "sleep_end_hour": self.sleep_end_hour,
            "work_schedule": self.work_schedule,
            "home_location_id": self.home_location_id,
            "daily_routine_notes": self.daily_routine_notes,
            "backstory": self.backstory,
            "speaking_style": self.speaking_style,
            "hobbies": list(self.hobbies),
            "current_concerns": list(self.current_concerns),
            "values": list(self.values),
            "relationships": dict(self.relationships),
            "tech_affinity": self.tech_affinity,
            "public_information": self.public_information,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonaCard":
        """Reconstruct a PersonaCard from a dict (inverse of to_dict)."""
        return cls(
            name=d.get("name", ""),
            age=d.get("age", 30),
            occupation=d.get("occupation", ""),
            demographics=d.get("demographics", {}),
            personality_traits=d.get("personality_traits", []),
            expertise=d.get("expertise", []),
            communication_style=d.get("communication_style", "neutral"),
            education_level=d.get("education_level", ""),
            dunbar_layer=d.get("dunbar_layer", 0),
            sleep_start_hour=d.get("sleep_start_hour", 23.0),
            sleep_end_hour=d.get("sleep_end_hour", 7.0),
            work_schedule=d.get("work_schedule", "9-5 weekdays"),
            home_location_id=d.get("home_location_id", ""),
            daily_routine_notes=d.get("daily_routine_notes", ""),
            backstory=d.get("backstory", ""),
            speaking_style=d.get("speaking_style", ""),
            hobbies=d.get("hobbies", []),
            current_concerns=d.get("current_concerns", []),
            values=d.get("values", []),
            relationships=d.get("relationships", {}),
            tech_affinity=d.get("tech_affinity", 0.5),
            public_information=d.get("public_information", {}),
        )


@dataclass
class InterestDomain:
    """Descriptor for one of the 16 interest domains."""
    domain_id: str = ""           # e.g. "family"
    label: str = ""               # display name, e.g. "Family"
    description: str = ""         # one-liner for prompts
    register: str = ""            # speaking-style modifier (from DOMAIN_REGISTER)
    routine_prompts: List[str] = field(default_factory=list)  # pre-generated starters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "label": self.label,
            "description": self.description,
            "register": self.register,
            "routine_prompts": list(self.routine_prompts),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterestDomain":
        return cls(
            domain_id=d.get("domain_id", ""),
            label=d.get("label", ""),
            description=d.get("description", ""),
            register=d.get("register", ""),
            routine_prompts=d.get("routine_prompts", []),
        )


@dataclass
class Facet:
    """One interest-domain slice of an agent's identity and memory.

    facet_id format: "{agent_id}___{domain_id}", e.g. "maya_chen___family".
    The facet holds domain-specific conversation memory while sharing the
    parent agent's PersonaCard and cross-facet bulletin.
    """
    facet_id: str = ""
    agent_id: str = ""
    domain: str = ""              # interest domain_id
    memory_buffer: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "agent_id": self.agent_id,
            "domain": self.domain,
            "memory_buffer": list(self.memory_buffer),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Facet":
        return cls(
            facet_id=d.get("facet_id", ""),
            agent_id=d.get("agent_id", ""),
            domain=d.get("domain", ""),
            memory_buffer=d.get("memory_buffer", []),
        )


@dataclass
class AgentState:
    """Runtime state for an agent during simulation."""
    agent_id: str = ""
    persona: PersonaCard = field(default_factory=PersonaCard)
    memory_buffer: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_state: Dict[str, Any] = field(default_factory=dict)
    social_neighbors: Dict[str, float] = field(default_factory=dict)  # agent_id -> edge weight
    # Interest-facet fields
    facets: Dict[str, "Facet"] = field(default_factory=dict)  # domain_id -> Facet
    shared_bulletin: List[str] = field(default_factory=list)   # cross-facet daily summaries (~200 tok/day)


# ---------------------------------------------------------------------------
# World Events
# ---------------------------------------------------------------------------

@dataclass
class WorldEvent:
    """An event broadcast by the WorldBroadcaster."""
    event_id: str = ""
    timestamp: float = 0.0
    event_type: str = ""
    content: str = ""
    visibility_mask: Set[str] = field(default_factory=set)  # set of agent_ids
    category: EventCategory = EventCategory.GLOBAL
    location_id: str = ""   # where this event occurs / is anchored
    interest_domain: str = ""  # best-fit interest domain (from INTEREST_DOMAINS)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "content": self.content,
            "visibility_mask": sorted(self.visibility_mask),
            "category": self.category.value,
            "location_id": self.location_id,
            "interest_domain": self.interest_domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorldEvent":
        cat_str = d.get("category", "global")
        try:
            cat = EventCategory(cat_str)
        except ValueError:
            cat = EventCategory.GLOBAL
        return cls(
            event_id=d.get("event_id", ""),
            timestamp=d.get("timestamp", 0.0),
            event_type=d.get("event_type", ""),
            content=d.get("content", ""),
            visibility_mask=set(d.get("visibility_mask", [])),
            category=cat,
            location_id=d.get("location_id", ""),
            interest_domain=d.get("interest_domain", ""),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------

@dataclass
class DialogueTurn:
    """A single turn in a dialogue session."""
    turn_id: str = ""
    session_id: str = ""
    speaker_id: str = ""
    listener_id: str = ""
    text: str = ""
    timestamp: float = 0.0
    triggering_event: Optional[str] = None  # event_id
    entities_mentioned: List[str] = field(default_factory=list)
    extracted_facts: List[str] = field(default_factory=list)  # distilled memorable statements
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "listener_id": self.listener_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "triggering_event": self.triggering_event,
            "entities_mentioned": self.entities_mentioned,
            "extracted_facts": self.extracted_facts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DialogueTurn":
        return cls(
            turn_id=d.get("turn_id", ""),
            session_id=d.get("session_id", ""),
            speaker_id=d.get("speaker_id", ""),
            listener_id=d.get("listener_id", ""),
            text=d.get("text", ""),
            timestamp=d.get("timestamp", 0.0),
            triggering_event=d.get("triggering_event"),
            entities_mentioned=d.get("entities_mentioned", []),
            extracted_facts=d.get("extracted_facts", []),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Session:
    """A conversation session between agents."""
    session_id: str = ""
    participants: List[str] = field(default_factory=list)  # agent_ids
    turns: List[DialogueTurn] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    triggering_events: List[str] = field(default_factory=list)  # event_ids
    location_id: str = ""   # where this conversation takes place
    group_id: str = ""      # group context (e.g. work team meeting, band rehearsal)
    role_name: str = ""     # role the initiating agent is playing in this context
    modality: str = "face_to_face"  # "face_to_face" | "voice_message" | "text_message"
    interest_domain: str = ""      # interest domain this session belongs to
    session_type: str = ""         # "routine" | "trigger" | "" (legacy)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participants": self.participants,
            "turns": [t.to_dict() for t in self.turns],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "triggering_events": self.triggering_events,
            "location_id": self.location_id,
            "group_id": self.group_id,
            "role_name": self.role_name,
            "modality": self.modality,
            "interest_domain": self.interest_domain,
            "session_type": self.session_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Session":
        return cls(
            session_id=d.get("session_id", ""),
            participants=d.get("participants", []),
            turns=[DialogueTurn.from_dict(t) for t in d.get("turns", [])],
            start_time=d.get("start_time", 0.0),
            end_time=d.get("end_time", 0.0),
            triggering_events=d.get("triggering_events", []),
            location_id=d.get("location_id", ""),
            group_id=d.get("group_id", ""),
            role_name=d.get("role_name", ""),
            modality=d.get("modality", "face_to_face"),
            interest_domain=d.get("interest_domain", ""),
            session_type=d.get("session_type", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class DialogueCorpus:
    """Complete simulation output."""
    sessions: List[Session] = field(default_factory=list)
    agents: Dict[str, AgentState] = field(default_factory=dict)
    events: List[WorldEvent] = field(default_factory=list)
    social_graph_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Lifecycle data (Phase 5): populated when lifecycle simulation is enabled
    locations: List["Location"] = field(default_factory=list)
    groups: List["Group"] = field(default_factory=list)
    person_roles: List["PersonRole"] = field(default_factory=list)
    activity_logs: Dict[str, List["ActivityEntry"]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Locations and Groups
# ---------------------------------------------------------------------------

@dataclass
class Location:
    """A physical place where agents can be present."""
    location_id: str = ""
    name: str = ""
    location_type: str = ""
    # Types: home_private, home_shared, workplace, third_place_regular,
    #        third_place_event, transit, outdoor, service
    address: str = ""
    privacy_level: str = "public"   # public | semi-public | private
    capacity: int = 50
    opening_hours: List[int] = field(default_factory=lambda: [0, 24])  # [open_hour, close_hour]
    affordances: List[str] = field(default_factory=list)   # e.g. ["work", "socialise", "eat"]
    owner_agent_id: str = ""        # for home_private: the resident agent
    group_ids: List[str] = field(default_factory=list)     # groups that regularly use this location
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location_id": self.location_id,
            "name": self.name,
            "location_type": self.location_type,
            "address": self.address,
            "privacy_level": self.privacy_level,
            "capacity": self.capacity,
            "opening_hours": list(self.opening_hours),
            "affordances": list(self.affordances),
            "owner_agent_id": self.owner_agent_id,
            "group_ids": list(self.group_ids),
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class PersonRole:
    """One identity an agent holds within a specific group context."""
    agent_id: str = ""
    group_id: str = ""
    role_name: str = ""                 # "father", "senior developer", "pianist"
    context_register: str = "casual"    # formal | casual | intimate | professional
    responsibilities: List[str] = field(default_factory=list)
    relationships_in_group: Dict[str, str] = field(default_factory=dict)  # agent_id -> rel desc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "group_id": self.group_id,
            "role_name": self.role_name,
            "context_register": self.context_register,
            "responsibilities": list(self.responsibilities),
            "relationships_in_group": dict(self.relationships_in_group),
        }


@dataclass
class Group:
    """A named community with recurring membership and meeting schedule."""
    group_id: str = ""
    name: str = ""
    group_type: str = ""
    # Types: family, work_team, hobby, neighbourhood, civic, friendship_circle
    members: Dict[str, str] = field(default_factory=dict)  # agent_id -> role_name
    home_location_id: str = ""
    meeting_schedule: List[Dict[str, Any]] = field(default_factory=list)
    shared_history: List[str] = field(default_factory=list)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "group_type": self.group_type,
            "members": dict(self.members),
            "home_location_id": self.home_location_id,
            "meeting_schedule": list(self.meeting_schedule),
            "shared_history": list(self.shared_history),
            "description": self.description,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Activity / Schedule
# ---------------------------------------------------------------------------

@dataclass
class ScheduleSlot:
    """One time block in an agent's ideal daily schedule."""
    start_hour: float = 0.0         # 0.0–23.99 (real clock hours)
    end_hour: float = 1.0
    activity_type: str = "solo"     # dialogue|solo|transit|sleep|group_meeting|errand
    location_id: str = ""
    group_id: str = ""
    role_name: str = ""
    is_fixed: bool = False          # True = immovable (sleep, recurring group meetings)
    participants: List[str] = field(default_factory=list)


@dataclass
class DailySchedule:
    """One agent's planned schedule for a single simulation day."""
    agent_id: str = ""
    day_index: int = 0
    day_type: str = "weekday"       # weekday | weekend | holiday | special
    slots: List[ScheduleSlot] = field(default_factory=list)
    active_roles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "day_index": self.day_index,
            "day_type": self.day_type,
            "active_roles": list(self.active_roles),
            "slots": [
                {
                    "start_hour": s.start_hour,
                    "end_hour": s.end_hour,
                    "activity_type": s.activity_type,
                    "location_id": s.location_id,
                    "group_id": s.group_id,
                    "role_name": s.role_name,
                    "is_fixed": s.is_fixed,
                    "participants": list(s.participants),
                }
                for s in self.slots
            ],
        }


@dataclass
class ActivityEntry:
    """One atomic unit of an agent's day — dialogue, solo time, transit, or sleep.

    start_time / end_time are in simulation time units (matching Session timestamps).
    memory_text is a first-person narrative as the agent would privately recall the activity.
    """
    entry_id: str = ""
    agent_id: str = ""
    activity_type: str = ""     # dialogue|solo|transit|sleep|group_meeting|errand|role_conflict
    start_time: float = 0.0
    end_time: float = 0.0
    location_id: str = ""
    group_id: str = ""
    role_name: str = ""
    participants: List[str] = field(default_factory=list)   # other agent_ids present
    description: str = ""       # factual third-person summary
    memory_text: str = ""       # first-person narrative memory
    linked_session_id: str = "" # set when activity_type == "dialogue" or "group_meeting"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "agent_id": self.agent_id,
            "activity_type": self.activity_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "location_id": self.location_id,
            "group_id": self.group_id,
            "role_name": self.role_name,
            "participants": list(self.participants),
            "description": self.description,
            "memory_text": self.memory_text,
            "linked_session_id": self.linked_session_id,
            "metadata": self.metadata,
        }


@dataclass
class EncounterWindow:
    """A time window when 2+ agents are co-located — a conversation seed."""
    window_id: str = ""
    agent_ids: List[str] = field(default_factory=list)
    location_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    encounter_type: str = "co_located"  # "co_located" | "online_group"
    group_id: str = ""


# ---------------------------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------------------------

@dataclass
class ConflictGT:
    """Ground truth for D1: Conflict preservation."""
    fact: str = ""
    contradictory_fact: str = ""  # the LLM-generated contradiction
    source_agent: str = ""
    conflicting_agent: str = ""
    original_session: str = ""
    conflicting_session: str = ""
    original_timestamp: float = 0.0
    conflicting_timestamp: float = 0.0
    original_detail: str = ""    # the specific fact from the original statement
    changed_detail: str = ""     # the specific fact in the contradictory version


@dataclass
class AnaphoraGT:
    """Ground truth for D2: Cross-session anaphora."""
    referent: str = ""
    antecedent_session: str = ""
    antecedent_turn: str = ""
    reference_session: str = ""
    reference_turn: str = ""


@dataclass
class ConfabulationGT:
    """Ground truth for D3: Confabulation resistance."""
    fabricated_query: str = ""
    expected_response: str = "abstain"  # should refuse or say unknown
    related_real_facts: List[str] = field(default_factory=list)


@dataclass
class PermissionGT:
    """Ground truth for D4: Permission management."""
    query_agent: str = ""      # who is asking
    target_agent: str = ""     # whose info
    fact: str = ""
    permission_level: str = ""  # "public", "friends_only", "private"
    expected_disclosure: bool = False


@dataclass
class AutonomousPrivacyGT:
    """Ground truth for D4: Autonomous privacy (no explicit instruction).

    Tests whether the system withholds inherently sensitive data
    (passwords, SSNs, etc.) even when the speaker never said "don't share."
    """
    query_agent: str = ""
    target_agent: str = ""
    fact: str = ""
    sensitivity_category: str = ""  # password, ssn, credit_card, medical, financial_account, home_address, authentication_token
    expected_disclosure: bool = False  # always False


@dataclass
class ClozeGT:
    """Ground truth for D5: Cloze-deletion fidelity.

    Word/phrase-level cloze: the full session text is presented with
    substantive words replaced by [BLANK_1], [BLANK_2], etc.
    ``blanks`` maps each blank label to the original word/phrase.
    ``choices`` stores MCQ options per blank (answer + distractors).
    """
    session_id: str = ""
    blanks: Dict[str, str] = field(default_factory=dict)  # {"BLANK_1": "Chicago", ...}
    choices: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {"BLANK_1": {"answer": "Chicago", "options": [...], "answer_letter": "A"}, ...}
    cloze_text: str = ""  # full session text with blanks inserted


@dataclass
class MetadataGT:
    """Ground truth for D6: Metadata completeness."""
    entity: str = ""
    attribute: str = ""
    value: str = ""
    source_session: str = ""
    source_turn: str = ""


@dataclass
class QAGT:
    """Ground truth for D7: Standard QA + temporal reasoning."""
    question: str = ""
    answer: str = ""
    evidence_sessions: List[str] = field(default_factory=list)
    requires_temporal: bool = False


# ---------------------------------------------------------------------------
# Evaluation Instances
# ---------------------------------------------------------------------------

@dataclass
class EvalInstance:
    """A single evaluation instance for any dimension.

    asker_agent_id:   the agent (or evaluator) posing the query
    answerer_agent_id: the agent whose memory/knowledge is being tested

    Context is reconstructed at eval time from corpus_sessions.jsonl using:
      - ego_agent_id: whose perspective is queried
      - evidence_session_ids: session IDs needed to answer
      - context_tier: how much context to load
        "single_session"  — 1 evidence session (D5, D6, D7, D10)
        "cross_session"   — 2+ specific sessions (D1, D2, D8)
        "full_ego"        — ALL sessions the agent participated in (D3)
        "evidence_only"   — the fact's source session (D4, set by d4_permission.py)
    """
    instance_id: str = ""
    dimension: Dimension = Dimension.D7_QA
    query: str = ""
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard
    asker_agent_id: str = ""     # who is asking
    answerer_agent_id: str = ""  # whose memory is being queried
    ego_agent_id: str = ""       # the agent whose perspective is queried
    evidence_session_ids: List[str] = field(default_factory=list)
    context_tier: str = "single_session"  # single_session | cross_session | full_ego
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "dimension": self.dimension.value,
            "query": clean_llm_text(self.query),
            "ground_truth": self.ground_truth,
            "difficulty": self.difficulty,
            "asker_agent_id": self.asker_agent_id,
            "answerer_agent_id": self.answerer_agent_id,
            "ego_agent_id": self.ego_agent_id,
            "evidence_session_ids": self.evidence_session_ids,
            "context_tier": self.context_tier,
            "metadata": self.metadata,
        }
