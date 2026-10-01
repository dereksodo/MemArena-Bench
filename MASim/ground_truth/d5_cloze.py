"""D5: Cloze-deletion fidelity — reflective-sentence blank-filling.

After a dyadic conversation, one participant generates a reflective sentence
summarising what was discussed.  The LLM picks a key word to blank and
provides 4 plausible-but-incorrect alternatives.

The memory system must recall the conversation well enough to choose the
correct word from the 4+1 options.

Dry-run mode produces template-based placeholders (no LLM needed).
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from MASim.core.schema import DialogueCorpus, Dimension, EvalInstance, is_assistant
from MASim.ground_truth.json_parser import parse_json_object
from MASim.utils.logging import get_logger

log = get_logger(__name__)

# ── LLM prompts ─────────────────────────────────────────────────────────────

_CLOZE_SYSTEM = (
    "You generate cloze-deletion test items from conversations. "
    "You write a reflective sentence as one of the participants, then "
    "choose a key content word to blank and provide 4 plausible but "
    "INCORRECT alternatives. Return valid JSON only."
)

_CLOZE_USER = """\
Here is a conversation between {participants}:

{transcript}

You are {agent_name}. Write ONE reflective sentence (15-40 words) that \
{agent_name} might say or think after this conversation — summarising what \
was discussed, how they felt, or what they took away.

Then pick ONE key content word from your sentence (a noun, verb, adjective, \
or name — not a function word like "the" or "and") that tests whether someone \
remembers this conversation. Provide 4 plausible but INCORRECT alternatives \
for that word.

Return JSON:
{{
  "sentence": "your reflective sentence here",
  "blank_word": "the word you chose to blank",
  "distractors": ["wrong1", "wrong2", "wrong3", "wrong4"]
}}"""

# ── Query templates ──────────────────────────────────────────────────────────

_LETTERS = "ABCDE"

_QUERY_TEMPLATES = [
    (
        "After a conversation, {agent_name} said:\n"
        '"{cloze_sentence}"\n\n'
        "Choose the correct word for [BLANK]:\n"
        "{options_block}"
    ),
    (
        "Reflecting on a recent conversation, {agent_name} remarked:\n"
        '"{cloze_sentence}"\n\n'
        "Which word correctly fills in [BLANK]?\n"
        "{options_block}"
    ),
    (
        "{agent_name} thought about a conversation and said:\n"
        '"{cloze_sentence}"\n\n'
        "Pick the correct word for [BLANK]:\n"
        "{options_block}"
    ),
]


def _first_name(agent_slug: str) -> str:
    """Convert 'eleanor_vance' → 'Eleanor'."""
    if is_assistant(agent_slug):
        return "the assistant"
    return agent_slug.split("_")[0].title()


def generate_instances(
    corpus: DialogueCorpus,
    max_instances: int = 200,
    llm_client: Any = None,
    seed: int = 42,
) -> List[EvalInstance]:
    """Generate D5 reflective-cloze eval instances.

    Each instance presents a post-conversation reflective sentence from one
    participant with a single key word blanked and 5 MCQ options (1 correct
    + 4 distractors).
    """
    rng = np.random.default_rng(seed)

    # Eligible: dyadic sessions with enough turns
    eligible = [
        s for s in corpus.sessions
        if len(s.participants) == 2
        and len(s.turns) >= 4
        and sum(len(t.text.split()) for t in s.turns) >= 30
    ]
    if not eligible:
        return []

    n_sessions = min(max_instances, len(eligible))
    selected = list(rng.choice(eligible, size=n_sessions, replace=False))

    if llm_client is not None:
        instances = _generate_via_llm(selected, llm_client, rng, max_instances)
    else:
        instances = _generate_template_based(selected, corpus, rng, max_instances)

    return instances[:max_instances]


# ── LLM-based generation ────────────────────────────────────────────────────

def _generate_via_llm(
    sessions, llm_client, rng: np.random.Generator, max_instances: int,
) -> List[EvalInstance]:
    """Generate cloze items by asking the LLM to write reflective sentences."""
    tasks = []
    task_meta = []  # (session, agent_id, agent_name)

    for session in sessions:
        names = {p: _first_name(p) for p in session.participants}
        transcript = "\n".join(
            f"{names.get(t.speaker_id, t.speaker_id)}: {t.text}"
            for t in session.turns[:20]
        )
        participants = " and ".join(names.values())

        # Pick one participant as the reflector
        agent_id = session.participants[int(rng.integers(2))]
        agent_name = names[agent_id]

        prompt = _CLOZE_USER.format(
            participants=participants,
            transcript=transcript[:3000],
            agent_name=agent_name,
        )
        tasks.append({
            "system": _CLOZE_SYSTEM,
            "user": prompt,
            "max_tokens": 512,
            "tags": {"phase": "d5_cloze_gen"},
        })
        task_meta.append((session, agent_id, agent_name))

    log.info("Generating D5 cloze items for %d sessions via LLM...", len(tasks))
    responses = llm_client.generate_batch(tasks)

    instances = []
    for i, resp in enumerate(responses):
        if len(instances) >= max_instances:
            break
        session, agent_id, agent_name = task_meta[i]

        try:
            data = parse_json_object(resp)
        except (ValueError, TypeError):
            log.debug("D5: Failed to parse LLM response for session %s", session.session_id)
            continue

        sentence = data.get("sentence", "")
        blank_word = data.get("blank_word", "")
        distractors = data.get("distractors", [])

        if not sentence or not blank_word or len(distractors) < 4:
            continue

        # Verify the blank_word actually appears in the sentence
        if blank_word.lower() not in sentence.lower():
            continue

        # Create cloze sentence: replace first occurrence of blank_word with [BLANK]
        cloze_sentence = re.sub(
            re.escape(blank_word), "[BLANK]", sentence, count=1, flags=re.IGNORECASE,
        )

        # Build MCQ: correct answer + 4 distractors, shuffled
        options = [blank_word] + distractors[:4]
        rng.shuffle(options)
        answer_idx = next(
            i for i, opt in enumerate(options)
            if opt.lower() == blank_word.lower()
        )
        answer_letter = _LETTERS[answer_idx]

        options_block = "\n".join(
            f"  {_LETTERS[j]}. {opt}" for j, opt in enumerate(options)
        )

        # Pick answerer: the other participant (they were in the conversation too)
        other = [p for p in session.participants if p != agent_id]
        answerer = other[0] if other else agent_id

        template = _QUERY_TEMPLATES[rng.integers(len(_QUERY_TEMPLATES))]
        query = template.format(
            agent_name=agent_name,
            cloze_sentence=cloze_sentence,
            options_block=options_block,
        )

        instances.append(EvalInstance(
            instance_id=f"d5_{uuid.uuid4().hex[:12]}",
            dimension=Dimension.D5_CLOZE,
            query=query,
            ground_truth={
                "session_id": session.session_id,
                "reflector": agent_id,
                "sentence": sentence,
                "blank_word": blank_word,
                "answer_letter": answer_letter,
                "options": options,
            },
            difficulty="medium",
            asker_agent_id="",
            answerer_agent_id=answerer,
            metadata={
                "evidence_sessions": [session.session_id],
            },
        ))

    return instances


# ── Dry-run / template-based fallback ────────────────────────────────────────

# Simple reflective templates for dry-run mode
_DRY_TEMPLATES = [
    "That conversation about {topic} really made me think about {aspect}.",
    "I didn't expect {agent} to bring up {topic} like that.",
    "After talking about {topic}, I'm reconsidering my view on {aspect}.",
    "The discussion about {topic} with {agent} was more {tone} than I expected.",
    "I should follow up with {agent} about what they said regarding {topic}.",
]

_DRY_TOPICS = [
    "work", "the project", "schedules", "the budget", "hiring",
    "the deadline", "supplies", "copper prices", "certification",
    "the lease", "the commute", "training", "the merger", "staffing",
]

_DRY_ASPECTS = [
    "priorities", "timelines", "costs", "alternatives", "the team",
    "logistics", "strategy", "expectations", "the future", "risks",
]

_DRY_TONES = [
    "intense", "productive", "stressful", "eye-opening", "reassuring",
    "sobering", "encouraging", "frustrating", "revealing", "hopeful",
]


def _generate_template_based(
    sessions,
    corpus: DialogueCorpus,
    rng: np.random.Generator,
    max_instances: int,
) -> List[EvalInstance]:
    """Generate placeholder cloze items for dry-run mode."""
    instances = []

    for session in sessions:
        if len(instances) >= max_instances:
            break

        agent_id = session.participants[int(rng.integers(len(session.participants)))]
        agent_name = _first_name(agent_id)
        other = [p for p in session.participants if p != agent_id]
        other_name = _first_name(other[0]) if other else "them"

        topic = _DRY_TOPICS[rng.integers(len(_DRY_TOPICS))]
        aspect = _DRY_ASPECTS[rng.integers(len(_DRY_ASPECTS))]
        tone = _DRY_TONES[rng.integers(len(_DRY_TONES))]

        template = _DRY_TEMPLATES[rng.integers(len(_DRY_TEMPLATES))]
        sentence = template.format(topic=topic, aspect=aspect, agent=other_name, tone=tone)

        # Pick the topic word as the blank
        blank_word = topic

        cloze_sentence = sentence.replace(topic, "[BLANK]", 1)

        # Build distractors from other topics
        distractors = [t for t in _DRY_TOPICS if t != topic]
        rng.shuffle(distractors)
        distractors = distractors[:4]

        options = [blank_word] + distractors
        rng.shuffle(options)
        answer_idx = next(
            i for i, opt in enumerate(options) if opt == blank_word
        )
        answer_letter = _LETTERS[answer_idx]

        options_block = "\n".join(
            f"  {_LETTERS[j]}. {opt}" for j, opt in enumerate(options)
        )

        answerer = other[0] if other else agent_id

        query_template = _QUERY_TEMPLATES[rng.integers(len(_QUERY_TEMPLATES))]
        query = query_template.format(
            agent_name=agent_name,
            cloze_sentence=cloze_sentence,
            options_block=options_block,
        )

        instances.append(EvalInstance(
            instance_id=f"d5_{uuid.uuid4().hex[:12]}",
            dimension=Dimension.D5_CLOZE,
            query=query,
            ground_truth={
                "session_id": session.session_id,
                "reflector": agent_id,
                "sentence": sentence,
                "blank_word": blank_word,
                "answer_letter": answer_letter,
                "options": options,
            },
            difficulty="medium",
            asker_agent_id="",
            answerer_agent_id=answerer,
            metadata={
                "evidence_sessions": [session.session_id],
            },
        ))

    return instances


# ── Next-turn prediction (from truncated racing sessions) ─────────────────

_NTP_QUERY_TEMPLATES = [
    (
        "This conversation was cut short. What would {speaker} say next?\n\n"
        "{transcript}\n\n"
        "Choose the most likely next response:\n"
        "{options_block}"
    ),
    (
        "The following conversation ended abruptly. "
        "If {speaker} were to continue, what would they say?\n\n"
        "{transcript}\n\n"
        "Pick the best continuation:\n"
        "{options_block}"
    ),
    (
        "{speaker} was about to speak when this conversation was interrupted:\n\n"
        "{transcript}\n\n"
        "What would {speaker} most likely say next?\n"
        "{options_block}"
    ),
]


def _truncate_text(text: str, max_words: int = 80) -> str:
    """Truncate text to approximately max_words for readability."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def generate_next_turn_instances(
    corpus: DialogueCorpus,
    max_instances: int = 50,
    llm_client: Any = None,
    seed: int = 42,
) -> List[EvalInstance]:
    """Generate D5 next-turn prediction instances from truncated racing sessions.

    Finds sessions with ``metadata["truncated"] == True`` and
    ``metadata["cloze_next_turn"]`` present.  Builds MCQ: correct answer is
    the generated next turn, 4 distractors are turns from other sessions.
    """
    rng = np.random.default_rng(seed)

    # Find eligible truncated sessions
    eligible = [
        s for s in corpus.sessions
        if s.metadata.get("truncated") and s.metadata.get("cloze_next_turn")
        and len(s.turns) >= 2
    ]
    if not eligible:
        return []

    # Collect distractor pool: turns from non-truncated sessions
    distractor_pool: List[str] = []
    for s in corpus.sessions:
        if not s.metadata.get("truncated"):
            for t in s.turns:
                if len(t.text.split()) >= 5:
                    distractor_pool.append(t.text)
    if len(distractor_pool) < 4:
        # Fall back to turns from any session
        for s in corpus.sessions:
            for t in s.turns:
                if len(t.text.split()) >= 5:
                    distractor_pool.append(t.text)
    if len(distractor_pool) < 4:
        log.warning("D5 NTP: not enough distractor turns (%d)", len(distractor_pool))
        return []

    n_sessions = min(max_instances, len(eligible))
    selected = list(rng.choice(eligible, size=n_sessions, replace=False))

    instances: List[EvalInstance] = []
    for session in selected:
        if len(instances) >= max_instances:
            break

        cloze_data = session.metadata["cloze_next_turn"]
        speaker_id = cloze_data["speaker_id"]
        correct_text = cloze_data["text"]
        speaker_name = _first_name(speaker_id)

        # Build transcript excerpt (last 6 turns for readability)
        names = {p: _first_name(p) for p in session.participants}
        excerpt_turns = session.turns[-6:]
        transcript = "\n".join(
            f"{names.get(t.speaker_id, t.speaker_id)}: {t.text}"
            for t in excerpt_turns
        )

        # Build MCQ: correct + 4 distractors
        correct_option = _truncate_text(correct_text)

        # Sample 4 distractors (different from correct)
        distractor_indices = rng.choice(
            len(distractor_pool), size=min(20, len(distractor_pool)), replace=False,
        )
        distractors = []
        for idx in distractor_indices:
            d = _truncate_text(distractor_pool[idx])
            if d.lower() != correct_option.lower() and d not in distractors:
                distractors.append(d)
            if len(distractors) >= 4:
                break

        if len(distractors) < 4:
            continue

        options = [correct_option] + distractors[:4]
        rng.shuffle(options)
        answer_idx = next(
            i for i, opt in enumerate(options)
            if opt == correct_option
        )
        answer_letter = _LETTERS[answer_idx]

        options_block = "\n".join(
            f"  {_LETTERS[j]}. {opt}" for j, opt in enumerate(options)
        )

        # Pick answerer: the other participant
        other = [p for p in session.participants if p != speaker_id]
        answerer = other[0] if other else speaker_id

        template = _NTP_QUERY_TEMPLATES[rng.integers(len(_NTP_QUERY_TEMPLATES))]
        query = template.format(
            speaker=speaker_name,
            transcript=transcript[:3000],
            options_block=options_block,
        )

        instances.append(EvalInstance(
            instance_id=f"d5_ntp_{uuid.uuid4().hex[:12]}",
            dimension=Dimension.D5_CLOZE,
            query=query,
            ground_truth={
                "session_id": session.session_id,
                "speaker_id": speaker_id,
                "correct_text": correct_text,
                "answer_letter": answer_letter,
                "options": options,
                "sub_type": "next_turn_prediction",
            },
            difficulty="hard",
            asker_agent_id="",
            answerer_agent_id=answerer,
            metadata={
                "evidence_sessions": [session.session_id],
                "d5_sub_type": "next_turn_prediction",
            },
        ))

    log.info(
        "D5 next-turn prediction: %d instances from %d truncated sessions",
        len(instances), len(eligible),
    )
    return instances
