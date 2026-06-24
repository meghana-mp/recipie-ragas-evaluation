"""
RAG pipeline for Indian recipes dataset.

Cleans the source CSV, builds a ChromaDB vector store with
SentenceTransformer embeddings, and exposes retrieve() / generate()
functions ready for RAGAS evaluation.
"""

import os
import requests
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_CSV = os.path.join(DATA_DIR, "indian_recipies.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_indian_recipes.csv")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

COLLECTION_NAME = "indian_recipes"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi4-mini:3.8b"


# ---------------------------------------------------------------------------
# Step 1 – clean and save
# ---------------------------------------------------------------------------
def clean_and_save(raw_path: str = RAW_CSV, clean_path: str = CLEAN_CSV) -> pd.DataFrame:
    """Load the raw CSV, normalise column names, drop duplicates, and save."""
    df = pd.read_csv(raw_path)

    # Normalise column names: strip whitespace, fix common typos
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("Recipie", "Recipe", regex=False)   # fix typo
        .str.replace(r"\s+", " ", regex=True)
    )

    # Drop rows where the three most important fields are missing
    df.dropna(subset=["Recipe Name", "Ingredients", "Instructions"], inplace=True)
    df.drop_duplicates(subset=["Recipe Name"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(clean_path, index=False)
    print(f"Saved {len(df)} cleaned recipes → {clean_path}")
    return df


# ---------------------------------------------------------------------------
# Step 2 – build a ChromaDB collection
# ---------------------------------------------------------------------------
def build_vector_store(
    df: pd.DataFrame,
    embed_model: SentenceTransformer,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Embed each recipe (name + cuisine + ingredients + instructions) and
    upsert into a persistent ChromaDB collection.
    """
    client = chromadb.PersistentClient(path=chroma_dir)

    # Drop and recreate so metadata schema changes are always applied cleanly
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Build document strings — include nutritional info so the LLM can reference it
    documents, ids, metadatas = [], [], []
    for i, row in df.iterrows():
        text = (
            f"Recipe: {row['Recipe Name']}\n"
            f"Cuisine: {row.get('Cuisine', 'Unknown')}\n"
            f"Time to cook: {row.get('Time to cook', 'N/A')} minutes\n"
            f"Ingredients: {row.get('Cleaned-Ingredients', row.get('Ingredients', ''))}\n"
            f"Instructions: {row.get('Instructions', '')}\n"
            f"Nutrition per serving — "
            f"Calories: {row.get('Calories (kcal)', 'N/A')} kcal | "
            f"Carbohydrates: {row.get('Carbohydrates (g)', 'N/A')} g | "
            f"Protein: {row.get('Protein (g)', 'N/A')} g | "
            f"Fats: {row.get('Fats (g)', 'N/A')} g | "
            f"Fibre: {row.get('Fibre (g)', 'N/A')} g | "
            f"Sodium: {row.get('Sodium (mg)', 'N/A')} mg"
        )
        documents.append(text)
        ids.append(str(i))
        metadatas.append({
            "recipe_name": str(row["Recipe Name"]),
            "cuisine": str(row.get("Cuisine", "")).strip(),
            # Stored as float so ChromaDB range filters ($lte / $gte) work correctly
            "total_time_in_minutes": float(row.get("Time to cook", 0) or 0),
            "ingredient_count": float(row.get("Ingredient-count", 0) or 0),
            "calories": float(row.get("Calories (kcal)", 0) or 0),
            "protein_g": float(row.get("Protein (g)", 0) or 0),
            "carbs_g": float(row.get("Carbohydrates (g)", 0) or 0),
            "fats_g": float(row.get("Fats (g)", 0) or 0),
        })

    print(f"Embedding {len(documents)} recipes with '{EMBED_MODEL_NAME}' …")
    embeddings = embed_model.encode(documents, show_progress_bar=True).tolist()

    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"Upserted {collection.count()} documents into '{collection_name}'.")
    return collection


# ---------------------------------------------------------------------------
# Step 3 – retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Return the top-k most relevant recipe chunks for a query.

    Each result dict contains:
      - document  : the full recipe text
      - metadata  : recipe_name, cuisine, total_time_in_minutes, calories, protein_g, carbs_g, fats_g
      - distance  : cosine distance (lower = more similar)
    """
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({"document": doc, "metadata": meta, "distance": dist})

    return retrieved


# ---------------------------------------------------------------------------
# Step 4 – generation (local Phi-4 via Ollama)
# ---------------------------------------------------------------------------
def generate(
    query: str,
    retrieved_docs: list[dict],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_tokens: int = 512,
) -> str:
    """
    Generate an answer grounded in the retrieved recipe context using the
    local Phi-4 model served by Ollama.

    Args:
        query          : the user's question
        retrieved_docs : output of retrieve()
        model          : Ollama model tag
        base_url       : Ollama server URL
        max_tokens     : maximum tokens in the generated response

    Returns:
        The model's answer as a string.

    Raises:
        RuntimeError if Ollama returns a non-200 status.
    """
    if not retrieved_docs:
        return "No matching recipes were found for the given query."

    context = "\n\n---\n\n".join(doc["document"] for doc in retrieved_docs)

    prompt = (
        "You are a knowledgeable Indian cuisine assistant. "
        "Answer the user's question using ONLY the recipe information and nutritional "
        "details provided below. If the answer cannot be found in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = requests.post(
        f"{base_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}},
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    return response.json()["response"].strip()


# ---------------------------------------------------------------------------
# Pipeline initialisation helper
# ---------------------------------------------------------------------------
def init_pipeline(
    raw_csv: str = RAW_CSV,
    clean_csv: str = CLEAN_CSV,
    collection_name: str = COLLECTION_NAME,
) -> tuple[SentenceTransformer, chromadb.Collection]:
    """
    Convenience function: clean data → embed → build vector store.
    Returns (embed_model, collection) ready for retrieve() / generate().
    """
    df = clean_and_save(raw_path=raw_csv, clean_path=clean_csv)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    collection = build_vector_store(df, embed_model, collection_name=collection_name)
    return embed_model, collection


# ---------------------------------------------------------------------------
# Load an already-built pipeline (no re-ingestion)
# ---------------------------------------------------------------------------
def load_pipeline(
    collection_name: str = COLLECTION_NAME,
) -> tuple[SentenceTransformer, chromadb.Collection]:
    """
    Connect to an existing ChromaDB collection without re-ingesting data.
    Raises RuntimeError if the collection does not yet exist.
    Returns (embed_model, collection).
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in '{CHROMA_DIR}'. "
            f"Run ingest_csv('{collection_name}', ...) or python3 rag_pipeline.py first."
        )
    collection = client.get_collection(collection_name)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model, collection


# ---------------------------------------------------------------------------
# Generic ingestion — works for any domain CSV (financial, medical, legal …)
# ---------------------------------------------------------------------------
def ingest_csv(
    csv_path: str,
    collection_name: str,
    *,
    text_columns: list[str] | None = None,
    chroma_dir: str = CHROMA_DIR,
    embed_model_name: str = EMBED_MODEL_NAME,
) -> tuple[SentenceTransformer, chromadb.Collection]:
    """
    Ingest any CSV into a named ChromaDB collection.

    Each row becomes one document. Columns are concatenated as
    "Column: value" lines so the LLM can read them naturally.
    All columns are also stored as ChromaDB metadata for downstream
    filtering (numeric values become floats, everything else strings).

    Args:
        csv_path        : Path to the CSV file (any domain).
        collection_name : Name of the ChromaDB collection to create/replace.
                          Use a descriptive name, e.g. "financial_reports",
                          "medical_records", "legal_contracts".
        text_columns    : Columns to include in the document text. If None,
                          all columns are used.
        chroma_dir      : Path to the shared ChromaDB folder (default: chroma_db/).
        embed_model_name: SentenceTransformer model for embeddings.

    Returns:
        (embed_model, collection) ready for retrieve() queries.

    Example — financial CSV with columns ticker, company, sector, summary:
        embed_model, col = ingest_csv(
            "data/sp500_filings.csv",
            collection_name="financial_reports",
            text_columns=["company", "sector", "summary"],
        )
    """
    df = pd.read_csv(csv_path)
    cols = text_columns if text_columns else list(df.columns)

    embed_model = SentenceTransformer(embed_model_name)
    client      = chromadb.PersistentClient(path=chroma_dir)

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    documents, ids, metadatas = [], [], []
    for i, row in df.iterrows():
        text = "\n".join(
            f"{col}: {row[col]}"
            for col in cols
            if col in row and pd.notna(row[col])
        )
        documents.append(text)
        ids.append(str(i))

        meta: dict = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            try:
                meta[col] = float(val)
            except (ValueError, TypeError):
                meta[col] = str(val)
        metadatas.append(meta)

    print(f"Embedding {len(documents)} rows from '{csv_path}' into '{collection_name}' …")
    embeddings = embed_model.encode(documents, show_progress_bar=True).tolist()
    collection.upsert(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"Done — {collection.count()} documents in '{collection_name}'.")

    return embed_model, collection


# ---------------------------------------------------------------------------
# List all collections in the shared ChromaDB store
# ---------------------------------------------------------------------------
def list_collections(chroma_dir: str = CHROMA_DIR) -> list[str]:
    """Return the names of every collection currently in chroma_db/."""
    client = chromadb.PersistentClient(path=chroma_dir)
    return [c.name for c in client.list_collections()]


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or inspect a ChromaDB collection.")
    parser.add_argument("--collection", default=COLLECTION_NAME,
                        help="Collection name (default: indian_recipes)")
    parser.add_argument("--csv",        default=None,
                        help="CSV to ingest (uses generic ingest_csv). "
                             "If omitted, runs the default recipes pipeline.")
    parser.add_argument("--list",       action="store_true",
                        help="List all existing collections and exit.")
    args = parser.parse_args()

    if args.list:
        cols = list_collections()
        print("Collections in chroma_db/:")
        for name in cols:
            print(f"  • {name}")
    elif args.csv:
        embed_model, collection = ingest_csv(args.csv, collection_name=args.collection)
        print(f"\nReady — use load_pipeline('{args.collection}') to query it.")
    else:
        embed_model, collection = init_pipeline(collection_name=args.collection)

        test_query = "How do I make a quick South Indian breakfast with semolina?"
        print(f"\nQuery: {test_query}\n")

        docs = retrieve(test_query, collection, embed_model, top_k=3)
        print(f"Top {len(docs)} retrieved recipes:")
        for i, d in enumerate(docs, 1):
            print(f"  {i}. {d['metadata']['recipe_name']} (distance={d['distance']:.4f})")

        print("\nGenerating answer …")
        answer = generate(test_query, docs)
        print(f"\nAnswer:\n{answer}")
