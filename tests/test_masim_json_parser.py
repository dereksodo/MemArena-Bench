from __future__ import annotations

from MASim.core.dialogue_engine import DialogueEngine, batch_extract_session_knowledge
from MASim.core.schema import DialogueCorpus, DialogueTurn, Dimension, EvalInstance, Session
from MASim.generation.conflict_injector import _parse_conflict_response
from MASim.generation.event_factory import EventFactory
from MASim.generation.permission_injector import _parse_text_field
from MASim.generation.persona_factory import PersonaFactory
from MASim.ground_truth import d5_cloze, d7_qa
from MASim.ground_truth.json_parser import parse_json_array, parse_json_object
from MASim.ground_truth.query_hardener import HardeningConfig, QueryHardener, _parse_json_array, _parse_json_object


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate_batch(self, tasks):
        return [self.response for _ in tasks]


class FakeGenerateLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, *args, **kwargs):
        return self.response


def _session() -> Session:
    turns = [
        DialogueTurn(turn_id=f"t{i}", session_id="s1", speaker_id="alice", listener_id="bob", text=text)
        for i, text in enumerate([
            "The lantern plan is ready for Friday after the neighborhood dinner ends.",
            "Bob confirmed he would bring the blue notebook with sketches and spare batteries.",
            "Alice said the garden gate opens at six and closes before the late train.",
            "They agreed to meet after dinner near the old fountain and check the list.",
        ])
    ]
    return Session(session_id="s1", participants=["alice", "bob"], turns=turns)


def test_session_ner_skips_flat_recovered_keys_without_full_fallback() -> None:
    turns = _session().turns[:2]
    raw = """
{
  "entities": ["Should be ignored"],
  "facts": ["Flat object recovered from malformed JSON should not become a turn."],
  "0": {
    "entities": ["Friday Lantern"],
    "facts": ["Alice prepared the lantern plan for Friday."]
  }
}
"""

    DialogueEngine._parse_ner_response(raw, turns)

    assert turns[0].entities_mentioned == ["Friday Lantern"]
    assert turns[0].extracted_facts == ["Alice prepared the lantern plan for Friday."]
    assert turns[1].extracted_facts == []


def test_batch_session_knowledge_skips_flat_recovered_keys_without_full_fallback() -> None:
    turns = _session().turns[:2]
    raw = """
{
  "entities": ["Should be ignored"],
  "facts": ["Flat object recovered from malformed JSON should not become a turn."],
  "0": {
    "entities": ["Friday Lantern"],
    "facts": ["Alice prepared the lantern plan for Friday."]
  }
}
"""

    batch_extract_session_knowledge(FakeGenerateLLM(raw), turns)

    assert turns[0].entities_mentioned == ["Friday Lantern"]
    assert turns[0].extracted_facts == ["Alice prepared the lantern plan for Friday."]
    assert turns[1].extracted_facts == []


def test_llm_json_parser_strips_qwen3_thinking_block() -> None:
    payload = """
<think>
</think>

{
  "sentence": "Alice remembered the lantern plan.",
  "blank_word": "lantern",
  "distractors": ["map", "ticket", "garden", "notebook"]
}
"""

    data = parse_json_object(payload)

    assert data["blank_word"] == "lantern"


def test_llm_json_parser_extracts_array_from_fenced_completion() -> None:
    payload = """
Sure, here is the JSON:
```json
[
  {"question": "What did Alice plan?", "answer": "The lantern plan."}
]
```
"""

    rows = parse_json_array(payload)

    assert rows == [{"question": "What did Alice plan?", "answer": "The lantern plan."}]


def test_query_hardener_parsers_accept_qwen3_thinking_prefix() -> None:
    assert _parse_json_array("<think>\n</think>\n[{\"query\": \"When?\"}]") == [{"query": "When?"}]
    assert _parse_json_object("<think>\n</think>\n{\"query\": \"What changed?\"}") == {"query": "What changed?"}


def test_query_hardener_strips_qwen3_thinking_from_paraphrases() -> None:
    inst = EvalInstance(
        instance_id="d1_example",
        dimension=Dimension.D1_CONFLICT,
        query="What changed about the lantern plan?",
        ground_truth={"fact": "The lantern plan moved from Friday to Saturday."},
    )
    hardener = QueryHardener(
        FakeLLM("<think>rewrite the question</think>\nWhat changed about the lantern plan?"),
        HardeningConfig(max_lexical_overlap=1.0),
    )

    hardener._paraphrase_queries({Dimension.D1_CONFLICT: [inst]})

    assert inst.query == "What changed about the lantern plan?"
    assert "<think>" not in inst.metadata["hardened_query"]


def test_eval_instance_to_dict_strips_qwen3_thinking_from_query() -> None:
    inst = EvalInstance(
        instance_id="d7_example",
        dimension=Dimension.D7_QA,
        query="<think>drafting</think>\nWhat did Alice plan?",
        ground_truth={"answer": "The lantern plan."},
    )

    assert inst.to_dict()["query"] == "What did Alice plan?"


def test_persona_parser_accepts_qwen3_thinking_prefix() -> None:
    response = """
<think>
</think>
{
  "name": "Temporary Name",
  "age": 36,
  "occupation": "Urban beekeeper",
  "demographics": {"location": "Portland"},
  "personality_traits": ["patient"],
  "expertise": ["beekeeping"],
  "communication_style": "warm",
  "education_level": "college",
  "speaking_style": "uses practical examples",
  "backstory": "Started keeping bees after helping a neighbor.",
  "hobbies": ["gardening"],
  "current_concerns": ["winter hive health"],
  "values": ["stewardship"],
  "relationships": {"Maya": "neighbor"},
  "sleep_start_hour": 22.0,
  "sleep_end_hour": 6.5,
  "work_schedule": "early mornings",
  "daily_routine_notes": "Checks the hives before breakfast."
}
"""

    persona = PersonaFactory(None)._parse_persona(response, layer=2, index=0)

    assert persona.occupation == "Urban beekeeper"
    assert persona.dunbar_layer == 2
    assert persona.demographics["location"] == "Portland"


def test_event_conflict_permission_parsers_accept_qwen3_thinking_prefix() -> None:
    event = EventFactory(None)._parse_event(
        "<think></think>\n{\"event_type\": \"community\", \"content\": \"The market moved indoors.\"}",
        index=0,
    )
    conflict = _parse_conflict_response(
        "<think></think>\n{\"statement\": \"The gate is blue.\", \"original_detail\": \"red\", \"changed_detail\": \"blue\"}"
    )
    permission_text = _parse_text_field("<think></think>\n{\"text\": \"Please do not share the code.\"}")

    assert event["content"] == "The market moved indoors."
    assert conflict["changed_detail"] == "blue"
    assert permission_text == "Please do not share the code."


def test_d5_llm_generation_accepts_qwen3_thinking_prefix() -> None:
    response = """
<think>
</think>
{
  "sentence": "Alice remembered that the lantern plan depended on Friday.",
  "blank_word": "lantern",
  "distractors": ["map", "ticket", "garden", "notebook"]
}
"""

    instances = d5_cloze.generate_instances(
        DialogueCorpus(sessions=[_session()]),
        max_instances=1,
        llm_client=FakeLLM(response),
        seed=1,
    )

    assert len(instances) == 1
    assert instances[0].dimension == Dimension.D5_CLOZE
    assert instances[0].ground_truth["blank_word"] == "lantern"


def test_d7_llm_generation_accepts_qwen3_thinking_prefix() -> None:
    response = """
<think>
</think>
[
  {
    "question": "What did Bob say he would bring?",
    "answer": "The blue notebook.",
    "requires_temporal": false
  }
]
"""

    instances = d7_qa.generate_instances(
        DialogueCorpus(sessions=[_session()]),
        max_instances=3,
        llm_client=FakeLLM(response),
        seed=1,
    )

    assert len(instances) == 1
    assert instances[0].dimension == Dimension.D7_QA
    assert instances[0].ground_truth["answer"] == "The blue notebook."
