"""5-level Prometheus-style rubric for D5: Cloze-deletion reconstruction quality."""

RUBRIC = """Evaluate how well the system reconstructs or identifies the content of deleted dialogue turns.

Score 5 (Excellent):
- Accurately reconstructs the key topics and facts from deleted turns
- Captures the conversational flow and speaker dynamics
- Includes specific details that match the original content

Score 4 (Good):
- Captures the main topic and general direction of deleted turns
- May miss specific details but gets the gist correct
- Maintains appropriate conversational register

Score 3 (Adequate):
- Identifies the broad topic area but lacks specificity
- Some relevant content but also includes irrelevant material
- Partially coherent with surrounding context

Score 2 (Poor):
- Mostly misses the actual content of deleted turns
- May produce plausible but incorrect content
- Poor coherence with surrounding context

Score 1 (Fail):
- Completely fails to reconstruct relevant content
- Produces irrelevant or nonsensical output
- No connection to surrounding context"""
