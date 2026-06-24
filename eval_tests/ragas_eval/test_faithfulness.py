"""
test_faithfulness.py — Pytest test for the Faithfulness metric via RAGAS.

PURPOSE
-------
Verifies that every claim in the RAG-generated answer is grounded in the
retrieved recipe context. A failing test means the pipeline is hallucinating
facts that aren't present in the source data.

HOW IT WORKS
------------
For each row in the golden dataset:
  1. Phi-4 Mini (local via Ollama) generates an answer from the retrieved context.
  2. RAGAS Faithfulness metric scores each claim against the context via Claude.
  3. Score = supported_claims / total_claims  (range 0–1).

RUN
---
    export ANTHROPIC_API_KEY=your_key_here   (or set in .env)
    pytest eval_tests/ragas_eval/test_faithfulness.py -v -s
    EVAL_SAMPLE_SIZE=5 pytest eval_tests/ragas_eval/test_faithfulness.py -v -s
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
from ragas.metrics import faithfulness                            # noqa: E402
from ragas.llms import LangchainLLMWrapper                       # noqa: E402
from langchain_anthropic import ChatAnthropic                    # noqa: E402
from retrieve import generate_with_phi4                          # noqa: E402

GOLDEN_CSV      = os.environ.get(
    "GOLDEN_CSV",
    os.path.join(PROJECT_ROOT, "golden_data_set", "indian_recipies_dataset.csv"),
)
RESULTS_CSV     = os.path.join(PROJECT_ROOT, "eval_results", "faithfulness_results.csv")
CLAUDE_MODEL    = "claude-sonnet-4-6"
SCORE_THRESHOLD = 0.5
SAMPLE_SIZE     = int(os.environ.get("EVAL_SAMPLE_SIZE", 9))

faithfulness.llm = LangchainLLMWrapper(ChatAnthropic(model=CLAUDE_MODEL))


def test_faithfulness() -> None:
    """Faithfulness — is every claim in the answer grounded in the retrieved context?"""
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

    print(f"\n  Running RAGAS Faithfulness on {total} samples …")
    dataset   = EvaluationDataset(samples=samples)
    result    = evaluate(dataset=dataset, metrics=[faithfulness])
    scores_df = result.to_pandas()

    rows = []
    for i, meta in enumerate(row_metadata):
        score = float(scores_df["faithfulness"].iloc[i])
        rows.append({
            "run_timestamp":    ts,
            "judge_model":      CLAUDE_MODEL,
            "question":         meta["question"],
            "answer":           meta["answer"],
            "contexts_preview": meta["contexts_preview"],
            "ground_truth":     meta["ground_truth"],
            "faithfulness":     score,
            "pass_fail":        score >= SCORE_THRESHOLD,
            "threshold":        SCORE_THRESHOLD,
        })

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

    valid = [r["faithfulness"] for r in rows if r["faithfulness"] == r["faithfulness"]]
    mean  = sum(valid) / len(valid) if valid else float("nan")

    print(f"\nfaithfulness mean : {mean:.3f}  (threshold: {SCORE_THRESHOLD})")
    print(f"results           : {RESULTS_CSV}")

    assert mean >= SCORE_THRESHOLD, (
        f"Mean faithfulness {mean:.3f} is below threshold {SCORE_THRESHOLD}."
    )
