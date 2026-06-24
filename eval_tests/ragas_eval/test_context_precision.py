"""
test_context_precision.py — Pytest test for the Context Precision metric via RAGAS.

PURPOSE
-------
Verifies that the chunks ChromaDB retrieves are actually useful for answering
the question. Poor context precision means the retriever is fetching off-topic
or loosely related recipe chunks that waste tokens and can mislead the generator.

HOW IT WORKS
------------
RAGAS ContextPrecision judges each retrieved chunk: does it contribute useful
information for producing the ground-truth answer? Score = relevant / total chunks.

WHY GROUND TRUTH (NOT GENERATED ANSWER)
-----------------------------------------
Relevance is judged against the expected answer, not the generated one.
This avoids penalising good context just because Phi-4 hallucinated,
and avoids rewarding irrelevant context that accidentally supported a wrong answer.

RUN
---
    export ANTHROPIC_API_KEY=your_key_here   (or set in .env)
    pytest eval_tests/ragas_eval/test_context_precision.py -v -s
    EVAL_SAMPLE_SIZE=5 pytest eval_tests/ragas_eval/test_context_precision.py -v -s
"""

import ast
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAGAS_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RAGAS_EVAL_DIR)

load_dotenv(Path(PROJECT_ROOT) / ".env")

from ragas import evaluate, EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.metrics import context_precision                       # noqa: E402
from ragas.llms import LangchainLLMWrapper                       # noqa: E402
from langchain_anthropic import ChatAnthropic                    # noqa: E402
from retrieve import generate_with_phi4                          # noqa: E402

GOLDEN_CSV      = os.environ.get(
    "GOLDEN_CSV",
    os.path.join(PROJECT_ROOT, "golden_data_set", "indian_recipies_dataset.csv"),
)
RESULTS_CSV     = os.path.join(PROJECT_ROOT, "eval_results", "context_precision_results.csv")
CLAUDE_MODEL    = "claude-sonnet-4-6"
SCORE_THRESHOLD = 0.5
SAMPLE_SIZE     = int(os.environ.get("EVAL_SAMPLE_SIZE", 9))

context_precision.llm = LangchainLLMWrapper(ChatAnthropic(model=CLAUDE_MODEL))


def test_context_precision() -> None:
    """Context Precision — are the retrieved contexts relevant for producing the correct answer?"""
    df    = pd.read_csv(GOLDEN_CSV).head(SAMPLE_SIZE).reset_index(drop=True)
    ts    = datetime.now(timezone.utc).isoformat()
    total = len(df)

    samples      = []
    row_metadata = []

    for i, row in df.iterrows():
        question     = str(row["question"])
        contexts     = ast.literal_eval(row["contexts"])
        ground_truth = str(row["ground_truth"])
        docs         = [{"document": ctx} for ctx in contexts]

        print(f"\n  [{i+1}/{total}] Generating answer …")
        try:
            answer = generate_with_phi4(question, docs)
        except Exception as exc:
            print(f"  ERROR generating: {exc}")
            answer = ""

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))
        row_metadata.append({
            "question":         question,
            "answer":           answer,
            "ground_truth":     ground_truth,
            "contexts_preview": contexts[0][:200] if contexts else "",
        })

    print(f"\n  Running RAGAS ContextPrecision on {total} samples …")
    dataset   = EvaluationDataset(samples=samples)
    result    = evaluate(dataset=dataset, metrics=[context_precision])
    scores_df = result.to_pandas()

    rows = []
    for i, meta in enumerate(row_metadata):
        score = float(scores_df["context_precision"].iloc[i])
        rows.append({
            "run_timestamp":     ts,
            "judge_model":       CLAUDE_MODEL,
            "question":          meta["question"],
            "answer":            meta["answer"],
            "contexts_preview":  meta["contexts_preview"],
            "ground_truth":      meta["ground_truth"],
            "context_precision": score,
            "pass_fail":         score >= SCORE_THRESHOLD,
            "threshold":         SCORE_THRESHOLD,
        })

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    valid = [r["context_precision"] for r in rows if r["context_precision"] == r["context_precision"]]
    mean  = sum(valid) / len(valid) if valid else float("nan")

    print(f"\ncontext_precision mean : {mean:.3f}  (threshold: {SCORE_THRESHOLD})")
    print(f"results                : {RESULTS_CSV}")

    assert mean >= SCORE_THRESHOLD, (
        f"Mean context_precision {mean:.3f} is below threshold {SCORE_THRESHOLD}. "
        "Check that ChromaDB retrieval is returning relevant recipe chunks."
    )
