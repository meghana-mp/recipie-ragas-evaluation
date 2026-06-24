"""
test_ragas_evaluation.py — Combined RAG Triad evaluation using the RAGAS library.

PURPOSE
-------
Runs all three RAG-Triad metrics together on the full golden dataset in a
single pytest session. Use this when you want a complete picture of pipeline
health in one command. The three standalone test files cover the same metrics
individually — use those when debugging a specific metric.

METRICS COVERED
---------------
  Context Precision  (test_context_relevance)
    → Are the retrieved chunks relevant for producing the ground-truth answer?
    → Low score = retrieval problem (ChromaDB returning off-topic content)

  Faithfulness  (test_groundedness)
    → Is every claim in the generated answer supported by the context?
    → Low score = hallucination problem (Phi-4 inventing facts)

  Answer Relevancy  (test_answer_relevance)
    → Does the generated answer address the question asked?
    → Low score = evasion problem (answer is on a tangent)

JUDGE MODEL
-----------
Claude Sonnet 4.6 via LangchainLLMWrapper — configured on the module-level
metric objects so all three metrics share the same Claude client.

WHY A SESSION FIXTURE
---------------------
All three test functions share the `ragas_scores` fixture with scope="session".
The expensive work (generate 9 answers + RAGAS scoring) runs ONCE per session
and the three test functions just read the results.

RUN
---
    export ANTHROPIC_API_KEY=your_key_here   (or set in .env)
    pytest eval_tests/ragas_eval/test_ragas_evaluation.py -v -s
    EVAL_SAMPLE_SIZE=5 pytest eval_tests/ragas_eval/test_ragas_evaluation.py -v -s
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

# Suppress DeprecationWarnings from RAGAS's transitional API — the old-style
# metric objects (faithfulness, answer_relevancy, context_precision) are
# deprecated in favour of InstructorLLM-based metrics in 0.4.x, but they
# still work correctly with LangchainLLMWrapper + Claude.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAGAS_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, RAGAS_EVAL_DIR)

load_dotenv(Path(PROJECT_ROOT) / ".env")

from ragas import evaluate, EvaluationDataset, SingleTurnSample              # noqa: E402
from ragas.metrics import faithfulness, context_precision, answer_relevancy  # noqa: E402
from ragas.llms import LangchainLLMWrapper                                   # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper                      # noqa: E402
from langchain_anthropic import ChatAnthropic                                # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings                      # noqa: E402
from retrieve import generate_with_phi4                                      # noqa: E402

GOLDEN_CSV      = os.environ.get(
    "GOLDEN_CSV",
    os.path.join(PROJECT_ROOT, "golden_data_set", "indian_recipies_dataset.csv"),
)
RESULTS_CSV     = os.path.join(PROJECT_ROOT, "eval_results", "ragas_results.csv")
CLAUDE_MODEL    = "claude-sonnet-4-6"
EMBED_MODEL     = "all-MiniLM-L6-v2"
SCORE_THRESHOLD = 0.5
SAMPLE_SIZE     = int(os.environ.get("EVAL_SAMPLE_SIZE", 9))

# Configure RAGAS metric objects once at module load. All three test functions
# share these so the Claude client is only instantiated once per session.
_llm        = LangchainLLMWrapper(ChatAnthropic(model=CLAUDE_MODEL))
_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))

faithfulness.llm            = _llm
context_precision.llm       = _llm
answer_relevancy.llm        = _llm
answer_relevancy.embeddings = _embeddings


@pytest.fixture(scope="session")
def ragas_scores() -> dict:
    """
    Generate answers with Phi-4, evaluate with RAGAS, and return mean scores.

    Scope is "session" so generation + scoring only runs once even though
    three test functions consume this fixture.
    """
    df    = pd.read_csv(GOLDEN_CSV).head(SAMPLE_SIZE).reset_index(drop=True)
    ts    = datetime.now(timezone.utc).isoformat()
    total = len(df)

    # Step 1: generate answers with Phi-4 and build RAGAS samples.
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

    # Step 2: run all three RAGAS metrics in one evaluate() call.
    print(f"\n  Running RAGAS evaluate() on {total} samples …")
    dataset = EvaluationDataset(samples=samples)
    result  = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
    )
    scores_df = result.to_pandas()

    # Step 3: merge metadata with scores and write CSV.
    rows = []
    for i, meta in enumerate(row_metadata):
        f_score  = float(scores_df["faithfulness"].iloc[i])
        cp_score = float(scores_df["context_precision"].iloc[i])
        ar_score = float(scores_df["answer_relevancy"].iloc[i])

        rows.append({
            "run_timestamp":          ts,
            "judge_model":            CLAUDE_MODEL,
            "question":               meta["question"],
            "answer":                 meta["answer"],
            "contexts_preview":       meta["contexts_preview"],
            "ground_truth":           meta["ground_truth"],
            "faithfulness":           f_score,
            "context_precision":      cp_score,
            "answer_relevancy":       ar_score,
            "faithfulness_pass":      f_score  >= SCORE_THRESHOLD,
            "context_precision_pass": cp_score >= SCORE_THRESHOLD,
            "answer_relevancy_pass":  ar_score >= SCORE_THRESHOLD,
            "threshold":              SCORE_THRESHOLD,
        })

    pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to: {RESULTS_CSV}")

    def _mean(key: str) -> float:
        vals = [r[key] for r in rows if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    means = {
        "faithfulness":      _mean("faithfulness"),
        "context_precision": _mean("context_precision"),
        "answer_relevancy":  _mean("answer_relevancy"),
    }
    print(f"  faithfulness      : {means['faithfulness']:.3f}")
    print(f"  context_precision : {means['context_precision']:.3f}")
    print(f"  answer_relevancy  : {means['answer_relevancy']:.3f}")
    return means


def test_context_relevance(ragas_scores: dict) -> None:
    """Context Precision — are the retrieved contexts relevant for producing the answer?"""
    score = ragas_scores["context_precision"]
    assert score >= SCORE_THRESHOLD, (
        f"Context precision {score:.3f} < threshold {SCORE_THRESHOLD}. "
        "ChromaDB may be returning off-topic recipe chunks."
    )


def test_groundedness(ragas_scores: dict) -> None:
    """Faithfulness — is every claim in the answer supported by the retrieved contexts?"""
    score = ragas_scores["faithfulness"]
    assert score >= SCORE_THRESHOLD, (
        f"Faithfulness {score:.3f} < threshold {SCORE_THRESHOLD}. "
        "Phi-4 may be hallucinating facts not present in the retrieved context."
    )


def test_answer_relevance(ragas_scores: dict) -> None:
    """Answer Relevancy — does the generated answer actually address the question asked?"""
    score = ragas_scores["answer_relevancy"]
    assert score >= SCORE_THRESHOLD, (
        f"Answer relevancy {score:.3f} < threshold {SCORE_THRESHOLD}. "
        "Generated answers may be drifting off-topic."
    )
