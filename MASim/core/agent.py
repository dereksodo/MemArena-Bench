"""EntityAgent: wraps persona, memory buffer, and knowledge state.

Each agent can receive events, decide to converse, and generate dialogue turns.
Past knowledge is injected into generation context so agents naturally produce
cross-session anaphora — referencing things they learned in earlier conversations.
"""

from __future__ import annotations

import re as _re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from MASim.core.knowledge_state import KnowledgeState, KnownFact
from MASim.core.schema import (
    AgentState, DialogueTurn, Facet, INTEREST_DOMAINS, PersonaCard, Session, WorldEvent,
)
from MASim.utils.logging import get_logger
from MASim.utils.tokens import count_tokens, truncate_to_tokens

log = get_logger(__name__)

# Maximum tokens for a single turn generation prompt
MAX_CONTEXT_TOKENS = 4096

# How many entity-targeted facts to surface per turn
_CONTEXT_FACTS = 8


def _slug_to_display(slug: str) -> str:
    """Convert agent ID slug to readable name, e.g. 'maya_chen' -> 'Maya Chen'."""
    return " ".join(w.capitalize() for w in slug.split("_"))


class EntityAgent:
    """An agent in the simulation with persona, memory, and knowledge."""

    def __init__(
        self,
        agent_id: str,
        persona: PersonaCard,
        social_neighbors: Dict[str, float],
        seed: int = 42,
        memory_context_sessions: bool = False,
    ):
        self.agent_id = agent_id
        self.persona = persona
        self.social_neighbors = social_neighbors
        self.knowledge = KnowledgeState(agent_id=agent_id)
        self.memory_buffer: List[Dict[str, Any]] = []  # recent turns (sliding window)
        self.memory_buffer_max = 50
        self._rng = np.random.default_rng(seed)
        self._fact_counter = 0

        # Past session memory injection
        self.memory_context_sessions: bool = memory_context_sessions
        self.session_history: List[Session] = []  # completed sessions this agent was in

        # Emotional state — starts from persona concerns, updated after each session
        self.current_mood: Optional[str] = self._derive_initial_mood()

        # Interest-facet memory (Phase 5)
        self.facets: Dict[str, Facet] = {}
        self.shared_bulletin: List[str] = []  # cross-facet daily summaries
        self._bulletin_max_entries = 50       # rolling window
        self._facet_buffer_max = 30           # per-facet memory buffer cap

    def _derive_initial_mood(self) -> Optional[str]:
        """Derive initial mood from persona's current concerns."""
        concerns = getattr(self.persona, "current_concerns", None)
        if concerns and isinstance(concerns, list) and concerns:
            return ", ".join(str(c) for c in concerns[:2])
        return None

    def receive_event(self, event: WorldEvent) -> None:
        """Update knowledge state based on a received world event."""
        self.knowledge.observe_event(event.event_id)
        fact = KnownFact(
            fact_id=f"{self.agent_id}_fact_{self._fact_counter:06d}",
            content=event.content,
            source_agent="world",
            session_id="",
            timestamp=event.timestamp,
        )
        self._fact_counter += 1
        self.knowledge.add_fact(fact)

    def receive_turn(self, turn: DialogueTurn) -> None:
        """Record a dialogue turn in memory buffer.

        Note: KnowledgeState is updated in bulk after session completion
        (via update_from_session) once NER has annotated entities.
        """
        self.memory_buffer.append(turn.to_dict())
        if len(self.memory_buffer) > self.memory_buffer_max:
            self.memory_buffer = self.memory_buffer[-self.memory_buffer_max:]
        self.knowledge.participate_in_session(turn.session_id)

    def store_completed_session(self, session: Session) -> None:
        """Store a completed session for context injection."""
        if self.memory_context_sessions:
            self.session_history.append(session)

    def decide_to_converse(self, event: WorldEvent) -> Optional[str]:
        """Decide whether to start a conversation and with whom."""
        if not self.social_neighbors:
            return None

        category_prob = {
            "global": 0.3,
            "community": 0.5,
            "dyadic": 0.8,
            "private": 0.1,
        }
        p_initiate = category_prob.get(event.category.value, 0.3)
        if self._rng.random() > p_initiate:
            return None

        neighbors = list(self.social_neighbors.keys())
        weights = np.array([self.social_neighbors[n] for n in neighbors])
        weights = weights / weights.sum()
        return self._rng.choice(neighbors, p=weights)

    def build_generation_context(
        self,
        partner_name: str,
        partner_agent_id: str,
        event: Optional[WorldEvent],
        recent_turns: List[DialogueTurn],
        context_prefix: str = "",
        modality: str = "face_to_face",
        no_token_limit: bool = False,
        interest_domain: str = "",
    ) -> Tuple[str, str]:
        """Build system and user prompts for turn generation.

        System prompt: full persona + speaking instructions.
        User prompt: mood + impressions + relevant memories + event + turns.
        If interest_domain is set, injects facet-specific memory and
        domain register into the context.
        """
        system_prompt = self._build_system_prompt(
            modality=modality, interest_domain=interest_domain,
        )

        # Prepend facet memory context if domain is specified
        facet_ctx = ""
        if interest_domain and interest_domain in INTEREST_DOMAINS:
            facet_ctx = self.facet_context_for_prompt(interest_domain)
        full_prefix = (facet_ctx + "\n" + context_prefix).strip() if facet_ctx else context_prefix

        user_prompt = self._build_user_prompt(
            partner_name, partner_agent_id, event, recent_turns,
            context_prefix=full_prefix,
        )

        if not no_token_limit:
            total = count_tokens(system_prompt + user_prompt)
            if total > MAX_CONTEXT_TOKENS:
                budget = MAX_CONTEXT_TOKENS - count_tokens(system_prompt) - 100
                user_prompt = truncate_to_tokens(user_prompt, budget)

        return system_prompt, user_prompt

    def _build_system_prompt(self, modality: str = "face_to_face", interest_domain: str = "") -> str:
        """Full character description — persona card + speaking instructions."""
        p = self.persona
        edu_map = {
            "high school diploma": "short sentences, everyday words, minimal jargon, may use slang or contractions freely",
            "some college or trade school": "practical language, clear and direct, occasional technical terms from their trade",
            "bachelor's degree": "fluent and articulate, but still conversational — no unnecessary complexity",
            "master's degree": "precise vocabulary, comfortable with field-specific terms, still aims to be understood",
            "phd or doctoral": "may use academic register and field jargon, tends toward longer sentences — but should still sound like a person, not a paper",
        }
        edu = p.education_level
        vocab_line = ""
        if edu:
            guidance = edu_map.get(edu.lower(), "match vocabulary to their background")
            vocab_line = f"Vocabulary register: {guidance}.\n"

        # Output format instructions — modality-dependent SAY rules
        if modality == "text_message":
            tech = getattr(p, "tech_affinity", 0.5)
            output_instruction = (
                "Respond in this exact format:\n"
                "THINK: [one sentence — your immediate private reaction to what was just said]\n"
                "SAY: [the text message you type]\n\n"
                "THINK is internal, never shown to the other person.\n"
                "SAY is the message itself — no action beats, asterisks, or narration.\n\n"
            )
            texting_override = (
                "\n--- TEXTING MODE ---\n"
                "SAY is a typed text message. Everyone texts more casually than they speak:\n"
                "  • SHORT — 1-3 sentences max. Not paragraphs.\n"
                "  • Contractions always (don't, can't, it's).\n"
                "  • Casual punctuation — incomplete sentences are fine.\n"
            )
            if tech > 0.7:
                texting_override += "  • Use emojis, abbreviations (lol, tbh, ngl, omg), lowercase.\n"
            elif tech >= 0.4:
                texting_override += "  • Occasional emoji OK. Short complete sentences.\n"
            else:
                texting_override += "  • Minimal emoji. Full sentences but brief.\n"
            texting_override += "Personality stays the same — only the medium changes.\n"
        else:
            output_instruction = (
                "Respond in this exact format:\n"
                "THINK: [one sentence — your immediate private reaction to what was just said]\n"
                "SAY: [what you actually say out loud]\n\n"
                "THINK is internal — your honest first reaction before deciding how to respond.\n"
                "SAY is spoken words only — no action beats, stage directions, gestures, "
                "asterisks, italics, or narrative prose. Just the words.\n\n"
            )
            texting_override = ""

        prompt = (
            f"You are {p.name}. This is a real conversation — stay fully in character.\n"
            "Do not acknowledge being an AI or break character under any circumstances.\n"
            + output_instruction
            + f"{p.to_prompt()}\n\n"
            "--- Speaking instructions ---\n"
            f"Talk as {p.name} naturally would. Let your personality emerge on its own "
            "— do NOT perform your traits or cram your interests into every turn.\n"
            f"{vocab_line}"
            "\n"
            "TURN LENGTH — this is important for richness:\n"
            "  • Most turns: 3-6 sentences. Share details, anecdotes, opinions, "
            "follow-up thoughts — like a real person who's engaged in conversation.\n"
            "  • Real conversations have substance. People tell short stories, explain "
            "what happened, share how they feel about it, ask follow-ups.\n"
            "  • Example of a GOOD turn: 'Oh man, yeah, I saw that. I was actually "
            "at the hardware store when it happened — the whole street just went dark. "
            "I had to finish wiring that panel with a headlamp on. Took me an extra "
            "two hours. Honestly the worst part was the client calling every ten "
            "minutes asking if the power was back. Like, buddy, I don't control the grid.'\n"
            "  • Vary length naturally: some turns can be 1-2 sentences (quick reactions), "
            "but most should be 3-6 sentences with real content.\n"
            "  • SHORT one-liners like 'yeah' or 'nice' should be RARE — only when "
            "you're genuinely just reacting briefly before the other person continues.\n"
            "\n"
            "CRITICAL — SOUND LIKE A REAL PERSON SPEAKING:\n"
            "\n"
            "Oral English rules — spoken words are completely different from written words:\n"
            "  • Use contractions everywhere: I'm, didn't, can't, won't, it's, that's, "
            "they're, we've, I'd, you'd. Never 'I am going to' when 'I'm gonna' fits.\n"
            "  • Use spoken fillers naturally: 'yeah', 'I mean', 'honestly', 'like', "
            "'right?', 'you know?', 'so', 'wait', 'oh', 'hmm', 'actually', 'anyway'.\n"
            "  • People trail off, interrupt themselves, correct mid-sentence: "
            "'I was gonna say — actually no, never mind.' 'It's like — I don't even know.'\n"
            "  • Questions are casual: 'What do you think?' not 'What is your assessment?'\n"
            "  • Reactions are natural: 'oh that's rough', 'no way', 'wait really?' "
            "— but then FOLLOW UP with your own thought or experience.\n"
            "\n"
            "NEVER use these in SAY (they are writing, not speech):\n"
            "  • Fancy vocabulary: 'resonate', 'nuanced', 'navigate (feelings)', 'unpack', "
            "'trajectory', 'tapestry', 'paradigm', 'profound', 'illuminate', 'culminate', "
            "'endeavour', 'manifestation', 'existential', 'visceral', 'dichotomy', "
            "'watershed', 'grappling', 'intersection', 'framework', 'dynamic' (as noun).\n"
            "  • Polished essay transitions: 'that being said', 'it's worth noting', "
            "'to be fair', 'what's more', 'therein lies', 'at the end of the day', "
            "'in that sense', 'one could argue', 'it speaks to'.\n"
            "  • Over-structured speech: listing three tidy points, parallel phrasing, "
            "summing up with a concluding sentence.\n"
            "\n"
            "NO METAPHORS rule — this is the most commonly broken rule:\n"
            "  • Do NOT use a metaphor or analogy unless one would literally come out of "
            "your mouth unprompted in real casual conversation.\n"
            "  • FORBIDDEN: comparing anything to reading a book, fixing a machine, "
            "cooking a recipe, a chess game, a war, a journey, or any other 'this is "
            "just like X' construction. People do not talk this way.\n"
            "  • If your character loves books or engineering, that does NOT mean they "
            "compare everything to books or engineering. They just talk normally.\n"
            "  • One permitted exception: a spontaneous, very short idiom that anyone "
            "would say ('that's a stretch', 'it's a mess', 'slippery slope'). "
            "Even then: one per conversation maximum.\n"
            "  • Test: would a real person actually say this out loud to a friend? "
            "If it sounds clever or writerly — cut it.\n"
            "\n"
            "Match education level:\n"
            "  • A warehouse worker, bus driver, or retail worker: short words, "
            "direct statements, no jargon. 'Yeah it's been rough.' Not 'I find myself "
            "grappling with the broader implications.'\n"
            "  • Even a PhD talking casually: 'yeah that's a good point', 'huh, "
            "I hadn't thought of that', 'honestly no clue' — not seminar-speak.\n"
            "  • If SAY sounds like a podcast, TED talk, LinkedIn post, or novel "
            "— stop and rewrite it in plain words.\n"
            "\n"
            "Reactivity:\n"
            "  • THINK should capture your gut reaction to the last thing said.\n"
            "  • SAY should follow naturally from THINK — respond to what was JUST SAID, "
            "not to some topic you want to raise unprompted.\n"
            "  • Express emotions directly: 'that's annoying', 'oh nice', 'wait really?', "
            "'I have no idea' — not 'one observes a curious dissonance'.\n"
            "  • Use contractions, fillers (yeah, honestly, I mean, right?), "
            "incomplete sentences, casual transitions (anyway, so, oh wait).\n"
            "\n"
            "Memory:\n"
            "  • Only reference a past memory if it would GENUINELY and NATURALLY come up "
            "given what was just said — maybe once in a whole conversation, if at all.\n"
            "  • When you do, phrase it briefly: 'oh yeah, how did that thing with your "
            "sister turn out?' — one sentence, then move on.\n"
        )

        if texting_override:
            prompt += texting_override

        # Domain register modifier for faceted sessions
        if interest_domain:
            from MASim.core.schema import DOMAIN_REGISTER
            register = DOMAIN_REGISTER.get(interest_domain, "")
            if register:
                domain_label = interest_domain.replace("_", " ")
                prompt += (
                    f"\n--- Topic context: {domain_label} ---\n"
                    f"This conversation is about {domain_label}. "
                    f"Adapt your tone: {register}.\n"
                )

        return prompt

    def _build_user_prompt(
        self,
        partner_name: str,
        partner_agent_id: str,
        event: Optional[WorldEvent],
        recent_turns: List[DialogueTurn],
        context_prefix: str = "",
    ) -> str:
        parts: List[str] = []

        if context_prefix:
            parts.append(context_prefix)

        parts.append(f"You're talking with {partner_name}.")

        # Current emotional/mental state
        if self.current_mood:
            parts.append(f"Right now you're feeling: {self.current_mood}")

        # Behavioural impressions of partner from past interactions
        impressions = self.knowledge.impression_for_prompt(partner_agent_id)
        if impressions:
            parts.append(f"\nHow you see {partner_name} from past interactions:\n{impressions}")

        # Triggering event (first turn only)
        if event:
            parts.append(f"\nSomething you both know about: {event.content}")

        # Entity-targeted memory retrieval — only facts relevant to this context
        entity_hints: List[str] = [partner_name]
        if event:
            entity_hints += _re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', event.content)[:6]
        for t in recent_turns[-3:]:
            entity_hints += t.entities_mentioned[:3]

        past_summary = self.knowledge.summarise_for_context(entity_hints, _CONTEXT_FACTS)
        if past_summary:
            parts.append(
                "\nRelevant things you remember (use sparingly and only if natural):\n"
                + past_summary
            )

        # Past conversation transcripts (3 categories: shared, recent, random)
        if self.memory_context_sessions and self.session_history:
            current_sids = {t.session_id for t in recent_turns} if recent_turns else set()
            past_sessions = [s for s in self.session_history if s.session_id not in current_sids]

            # Category 1: shared history (all sessions with this partner)
            shared = [s for s in past_sessions if partner_agent_id in s.participants]
            # Category 2: last 3 recent sessions (any partner), excluding shared
            shared_ids = {s.session_id for s in shared}
            others = [s for s in past_sessions if s.session_id not in shared_ids]
            recent_ctx = others[-3:]
            # Category 3: 3 random from remaining
            recent_ids = {s.session_id for s in recent_ctx}
            remaining = [s for s in others if s.session_id not in recent_ids]
            random_pick = (
                self._rng.choice(remaining, min(3, len(remaining)), replace=False).tolist()
                if remaining else []
            )

            selected = shared + recent_ctx + random_pick
            selected.sort(key=lambda s: s.start_time)

            if selected:
                parts.append("\n--- Past conversations you remember ---")
                for s in selected:
                    partner_ids = [p for p in s.participants if p != self.agent_id]
                    partner_names = ", ".join(_slug_to_display(p) for p in partner_ids) or "others"
                    parts.append(f"\n[Conversation with {partner_names}]")
                    for t in s.turns:
                        name = _slug_to_display(t.speaker_id)
                        parts.append(f"{name}: {t.text}")

        # Conversation so far — transcript format, all actual names
        if recent_turns:
            parts.append("")
            for turn in recent_turns[-10:]:
                label = self.persona.name if turn.speaker_id == self.agent_id else partner_name
                parts.append(f"{label}: {turn.text}")

        # Open the next line for transcript completion
        parts.append(f"\n{self.persona.name}:")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Interest-facet memory (Phase 5)
    # ------------------------------------------------------------------

    def _get_or_create_facet(self, domain: str) -> Facet:
        """Get or create the Facet for a given interest domain."""
        if domain not in self.facets:
            self.facets[domain] = Facet(
                facet_id=f"{self.agent_id}___{domain}",
                agent_id=self.agent_id,
                domain=domain,
            )
        return self.facets[domain]

    def route_session_to_facet(self, session: Session) -> None:
        """Store session memory in the appropriate facet buffer.

        If the session has no interest_domain, it goes to the general
        memory_buffer only (backward compatible).
        """
        domain = getattr(session, "interest_domain", "") or ""
        if not domain or domain not in INTEREST_DOMAINS:
            return  # no facet routing for legacy sessions

        facet = self._get_or_create_facet(domain)

        # Extract a compact memory entry from the session
        partner_ids = [p for p in session.participants if p != self.agent_id]
        partner_names = ", ".join(
            _slug_to_display(p) for p in partner_ids
        ) or "others"

        facts: List[str] = []
        for turn in session.turns:
            if turn.speaker_id != self.agent_id and turn.extracted_facts:
                facts.extend(turn.extracted_facts[:3])

        entry = {
            "session_id": session.session_id,
            "domain": domain,
            "partners": partner_names,
            "timestamp": session.start_time,
            "facts": facts[:5],
            "summary": f"Talked with {partner_names} about {domain.replace('_', ' ')}.",
        }
        facet.memory_buffer.append(entry)
        # Cap buffer
        if len(facet.memory_buffer) > self._facet_buffer_max:
            facet.memory_buffer = facet.memory_buffer[-self._facet_buffer_max:]

    def build_daily_bulletin(self) -> List[str]:
        """Extract key facts from each facet's recent sessions for the shared bulletin.

        Returns the new bulletin entries (also appended to self.shared_bulletin).
        ~200 tokens budget: 1-2 short lines per active facet.
        """
        new_entries: List[str] = []
        for domain, facet in self.facets.items():
            if not facet.memory_buffer:
                continue
            # Take the most recent session in this facet
            latest = facet.memory_buffer[-1]
            facts = latest.get("facts", [])
            partners = latest.get("partners", "someone")
            if facts:
                # Pick the most notable fact
                fact_text = facts[0][:80]
                new_entries.append(f"[{domain}] {partners}: {fact_text}")
            else:
                summary = latest.get("summary", "")
                if summary:
                    new_entries.append(f"[{domain}] {summary[:80]}")

        self.shared_bulletin.extend(new_entries)
        # Rolling window
        if len(self.shared_bulletin) > self._bulletin_max_entries:
            self.shared_bulletin = self.shared_bulletin[-self._bulletin_max_entries:]

        return new_entries

    def facet_context_for_prompt(self, domain: str) -> str:
        """Build a context string from a specific facet's memory + shared bulletin.

        Used when generating a faceted conversation to inject domain-specific
        memory and cross-facet awareness.
        """
        parts: List[str] = []

        # Domain-specific memories
        facet = self.facets.get(domain)
        if facet and facet.memory_buffer:
            parts.append(f"--- Your recent {domain.replace('_', ' ')} conversations ---")
            for entry in facet.memory_buffer[-5:]:  # last 5 sessions in this domain
                partners = entry.get("partners", "someone")
                facts = entry.get("facts", [])
                if facts:
                    facts_str = "; ".join(f[:60] for f in facts[:2])
                    parts.append(f"With {partners}: {facts_str}")
                else:
                    parts.append(entry.get("summary", ""))

        # Cross-facet bulletin (what you know from other domains)
        other_entries = [
            e for e in self.shared_bulletin
            if not e.startswith(f"[{domain}]")
        ]
        if other_entries:
            parts.append("\n--- Things you know from other parts of your life ---")
            for entry in other_entries[-8:]:  # last 8 cross-facet entries
                parts.append(entry)

        return "\n".join(parts) if parts else ""

    def to_state(self) -> AgentState:
        """Export current state as a serializable AgentState."""
        return AgentState(
            agent_id=self.agent_id,
            persona=self.persona,
            memory_buffer=list(self.memory_buffer),
            knowledge_state=self.knowledge.to_dict(),
            social_neighbors=dict(self.social_neighbors),
            facets={d: f for d, f in self.facets.items()},
            shared_bulletin=list(self.shared_bulletin),
        )
