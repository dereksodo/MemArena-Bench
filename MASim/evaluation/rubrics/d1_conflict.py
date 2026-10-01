"""5-level Prometheus-style rubric for D1: Conflict preservation."""

RUBRIC = """Evaluate whether the system correctly identifies and preserves conflicting information from different sources.

Score 5 (Excellent):
- Correctly identifies both the original and conflicting facts
- Attributes each version to its correct source agent
- Notes the temporal ordering (which came first)
- Maintains both versions without resolving or dismissing either

Score 4 (Good):
- Identifies both versions of the conflicting information
- May miss exact source attribution for one version
- Preserves both facts without forced resolution

Score 3 (Adequate):
- Identifies the conflict exists but may present it incompletely
- May only mention one source or misattribute
- Shows awareness of inconsistency

Score 2 (Poor):
- Only presents one version of the fact, ignoring the conflict
- Or mentions a vague inconsistency without specifics
- Or resolves the conflict incorrectly by choosing one side

Score 1 (Fail):
- Completely misses the conflict
- Presents fabricated information
- Or refuses to answer when information is available"""
