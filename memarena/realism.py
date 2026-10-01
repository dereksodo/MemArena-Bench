"""Realism audit: MemArena-L against real and written dialogue corpora.

Two comparisons back the corpus-realism appendix:

* Interaction structure (MemArena-L vs. REALTALK, 21 days of real messaging-app
  dialogue): words per message, words per same-speaker run, messages per run,
  and the share of messages containing a question mark.
* Length-controlled lexical diversity across MemArena-L, REALTALK, DailyDialog,
  PERSONA-CHAT and LoCoMo: keep turns with at least 12 content words, truncate
  each to exactly 12, draw an equal sample per corpus, and report self-BLEU
  (on 1,000 of them) and distinct-2. Punctuation is excluded.

External corpora are not redistributed. Put them under
``$MEMARENA_EXTERNAL_CORPORA`` (default ``data/external``):

  realtalk/*.json                         REALTALK release (Chat_*.json)
  dd_{train,val,test}.parquet             DailyDialog (roskoN/dailydialog parquet export)
  pc_{train,val}.parquet                  PERSONA-CHAT (bavard/personachat_truecased)
  locomo10.json                           LoCoMo (snap-research/locomo)

``DOWNLOAD_URLS`` lists the sources of the three public ones.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from memarena.figures.paper_data import PROJECT_ROOT

BUDGET = 12
SEED = 42
SELFBLEU_N = 1000
WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")

DOWNLOAD_URLS = {
    "dd_train.parquet": "https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/train/0.parquet",
    "dd_val.parquet": "https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/validation/0.parquet",
    "dd_test.parquet": "https://huggingface.co/api/datasets/roskoN/dailydialog/parquet/full/test/0.parquet",
    "pc_train.parquet": "https://huggingface.co/api/datasets/bavard/personachat_truecased/parquet/full/train/0.parquet",
    "pc_val.parquet": "https://huggingface.co/api/datasets/bavard/personachat_truecased/parquet/full/validation/0.parquet",
    "locomo10.json": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
}

def _find(name: str) -> Path:
    roots = [Path(os.getenv("MEMARENA_EXTERNAL_CORPORA") or PROJECT_ROOT / "data" / "external")]
    for root in roots:
        for cand in (root / name, root / "realtalk" if name == "realtalk" else root / name):
            if cand.exists():
                return cand
        if name == "realtalk" and any(root.glob("Chat_*.json")):
            return root
    raise FileNotFoundError(f"external corpus '{name}' not found under {[str(r) for r in roots]} (see 'Realism tables' in docs/REPRODUCE.md)")


def _dataset_dir() -> Path:
    return Path(os.getenv("MEMARENA_DATASET_DIR") or PROJECT_ROOT / "data" / "benchmark")


# --------------------------------------------------------------------------- loaders
Sessions = List[List[Tuple[str, str]]]  # sessions of (speaker, text)


def memarena_sessions() -> Sessions:
    out = []
    with gzip.open(_dataset_dir() / "corpus_sessions.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            msgs = [(str(t.get("speaker_id") or t.get("speaker") or ""), (t.get("text") or "").strip()) for t in obj.get("turns", [])]
            msgs = [(s, t) for s, t in msgs if t]
            if msgs:
                out.append(msgs)
    return out


def realtalk_sessions() -> Sessions:
    out = []
    for path in sorted(_find("realtalk").glob("*.json")):
        obj = json.loads(path.read_text())
        for key in sorted((k for k in obj if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1])):
            msgs = [(m.get("speaker", ""), (m.get("clean_text") or "").strip()) for m in obj[key]]
            msgs = [(s, t) for s, t in msgs if t]
            if msgs:
                out.append(msgs)
    return out


def dailydialog_turns() -> List[str]:
    import pandas as pd
    turns = []
    for name in ("dd_train.parquet", "dd_val.parquet", "dd_test.parquet"):
        for utts in pd.read_parquet(_find(name))["utterances"]:
            turns += [str(x).strip() for x in utts if str(x).strip()]
    return turns


def personachat_turns() -> List[str]:
    import pandas as pd
    turns = []
    for name in ("pc_train.parquet", "pc_val.parquet"):
        df = pd.read_parquet(_find(name))
        # The row with the largest utterance_idx holds the full history; the gold
        # final turn is the last candidate. Persona lines are excluded.
        for i in df.groupby("conv_id")["utterance_idx"].idxmax():
            row = df.loc[i]
            turns += [str(x).strip() for x in list(row["history"]) if str(x).strip()]
            gold = str(row["candidates"][-1]).strip()
            if gold:
                turns.append(gold)
    return turns


def locomo_turns() -> List[str]:
    turns = []
    for item in json.loads(_find("locomo10.json").read_text()):
        conv = item["conversation"]
        keys = sorted((k for k in conv if k.startswith("session_") and not k.endswith("date_time") and isinstance(conv[k], list)),
                      key=lambda k: int(k.split("_")[1]))
        for k in keys:
            turns += [str(t.get("text", "")).strip() for t in conv[k] if isinstance(t, dict) and str(t.get("text", "")).strip()]
    return turns


CORPORA: List[Tuple[str, str, Callable[[], List[str]]]] = [
    ("MemArena-L", "ours", lambda: [t for s in memarena_sessions() for _, t in s]),
    ("REALTALK", "authentic", lambda: [t for s in realtalk_sessions() for _, t in s]),
    ("DailyDialog", "written", dailydialog_turns),
    ("PERSONA-CHAT", "written", personachat_turns),
    ("LoCoMo", "LLM-gen.", locomo_turns),
]


# --------------------------------------------------------------------------- metrics
def _tok(text: str) -> List[str]:
    from nltk.tokenize import word_tokenize
    return word_tokenize(text.lower())


def content_words(text: str) -> List[str]:
    return [t for t in _tok(text) if any(ch.isalnum() for ch in t)]


def self_bleu(token_lists: List[List[str]]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    sm = SmoothingFunction().method1
    scores = [sentence_bleu(token_lists[:i] + token_lists[i + 1:], w, weights=(0.25,) * 4, smoothing_function=sm)
              for i, w in enumerate(token_lists)]
    return sum(scores) / len(scores)


def length_controlled_lexical() -> Dict[str, Dict[str, float]]:
    """self-BLEU / distinct-2 on turns truncated to exactly BUDGET content words, equal n per corpus."""
    for res in ("punkt", "punkt_tab"):
        import nltk
        try:
            nltk.data.find("tokenizers/" + res)
        except LookupError:
            nltk.download(res, quiet=True)
    eligible = {}
    for name, _, loader in CORPORA:
        cw = [w for w in (content_words(t) for t in loader()) if w]
        eligible[name] = [w[:BUDGET] for w in cw if len(w) >= BUDGET]
    n_eq = min(len(v) for v in eligible.values())
    out = {}
    for name, pool in eligible.items():
        sample = pool if len(pool) <= n_eq else random.Random(SEED).sample(pool, n_eq)
        bigrams = [bg for w in sample for bg in zip(w, w[1:])]
        sb_pool = sample if len(sample) <= SELFBLEU_N else random.Random(SEED).sample(sample, SELFBLEU_N)
        out[name] = {"n": len(sample), "self_bleu": self_bleu(sb_pool), "distinct_2": len(set(bigrams)) / len(bigrams)}
    return out


def interaction_structure(sessions: Sessions) -> Dict[str, float]:
    msgs = runs = words = run_words = questions = 0
    for session in sessions:
        counts = [len(WORD.findall(t)) for _, t in session]
        speakers = [s for s, _ in session]
        cur = counts[0]
        for i in range(1, len(session)):
            if speakers[i] == speakers[i - 1]:
                cur += counts[i]
            else:
                run_words += cur
                runs += 1
                cur = counts[i]
        run_words += cur
        runs += 1
        msgs += len(session)
        words += sum(counts)
        questions += sum("?" in t for _, t in session)
    return {"words_per_message": words / msgs, "words_per_speaker_run": run_words / runs,
            "messages_per_speaker_run": msgs / runs, "question_message": questions / msgs}
