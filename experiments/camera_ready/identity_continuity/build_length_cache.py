"""Build the per-session Qwen-tokenizer length cache that e2_structural.py's default
--length-metric qwen needs. Cheap (about 20s) but not derivable from the repo alone, so it
lives here rather than in a scratch directory.

  python3 build_length_cache.py            # writes $MEMARENA_TOKEN_CACHE or /tmp/session_tokens.json
"""
import json, gzip, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from e2_structural import BASE, CACHE, serialize          # same serialization, one source
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ.get('MEMARENA_TOKENIZER', 'Qwen/Qwen3-8B'))
out, t0 = {}, time.time()
with gzip.open(BASE / 'corpus_sessions.jsonl.gz', 'rt') as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        out[r['session_id']] = len(tok.encode(serialize(r), add_special_tokens=False))
        if (i + 1) % 4000 == 0:
            print(f'  {i+1} sessions, {time.time()-t0:.0f}s', flush=True)
Path(CACHE).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(CACHE, 'w'))
print(f'wrote {CACHE}: {len(out)} sessions in {time.time()-t0:.0f}s')
