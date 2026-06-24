"""
Filtered retrieval and local generation for the Indian recipes RAG pipeline.

retrieve_with_filters() queries ChromaDB with optional filters on cuisine,
cooking time, calorie count, and ingredient count. All filters are
auto-detected from natural language query text when not explicitly supplied.

generate_with_phi4() uses the local Ollama phi4-mini model to answer
questions grounded in the retrieved recipe context.
"""

import re
import requests
import chromadb
from sentence_transformers import SentenceTransformer

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi4-mini:3.8b"


# ---------------------------------------------------------------------------
# Internal: auto-detect cuisine from query
# ---------------------------------------------------------------------------
def _detect_cuisine_from_query(
    query: str,
    collection: chromadb.Collection,
) -> list[str] | None:
    """
    Scan the query for any cuisine name that exists in the collection.

    Matching is case-insensitive and also strips common suffixes like
    " Recipes" so that e.g. "South Indian" matches "South Indian Recipes".

    Shorter keywords that appear only as part of a longer matched keyword
    are suppressed — e.g. "Indian" is dropped when "South Indian Recipes"
    already matched.

    Returns a list of matching stored cuisine strings, or None if no match.
    """
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    stored_cuisines = {m["cuisine"] for m in all_meta if m.get("cuisine")}

    query_lower = query.lower()

    cuisine_to_kw: dict[str, str] = {}
    for cuisine in stored_cuisines:
        candidates = {cuisine.lower()}
        for suffix in (" recipes", " recipe"):
            if cuisine.lower().endswith(suffix):
                candidates.add(cuisine.lower()[: -len(suffix)].strip())

        matching = [
            kw for kw in candidates
            if re.search(r'\b' + re.escape(kw) + r'\b', query_lower)
        ]
        if matching:
            cuisine_to_kw[cuisine] = max(matching, key=len)

    if not cuisine_to_kw:
        return None

    all_kws = list(cuisine_to_kw.values())
    final = [
        cuisine for cuisine, kw in cuisine_to_kw.items()
        if not any(kw != other and kw in other for other in all_kws)
    ]
    return final if final else None


# ---------------------------------------------------------------------------
# Internal: auto-detect numeric filters from query
# ---------------------------------------------------------------------------
def _parse_filters_from_query(query: str) -> dict:
    """
    Extract numeric filter constraints from a natural language query.

    Recognised patterns
    -------------------
    Time       : "under/less than/within/at most N min[utes]"
                 "at least/more than/over N min[utes]"
                 "between N and M min[utes]"
                 "N min[utes] or less / or more"

    Calories   : "under/less than/at most N cal[ories]/kcal"
                 "at least/more than/over N cal[ories]/kcal"
                 "between N and M cal[ories]/kcal"
                 "low[- ]calorie" → max_calories=200

    Ingredients: "under/less than/fewer than/at most N ingredient[s]"
                 "at least/more than/over N ingredient[s]"
                 "between N and M ingredient[s]"

    Returns a dict with any of:
        min_time_in_minutes, max_time_in_minutes,
        min_calories, max_calories,
        min_ingredient_count, max_ingredient_count
    """
    q = query.lower()
    filters: dict = {}

    _upper = r"(?:under|less than|within|at most|max(?:imum)?|no more than|fewer than)"
    _lower = r"(?:at least|more than|over|minimum|min(?:imum)?)"
    _time  = r"(?:minutes?|mins?)"
    _cal   = r"(?:calories?|kcal|cal)"
    _ing   = r"ingredients?"
    _num   = r"(\d+(?:\.\d+)?)"

    # ---- Time ----
    if m := re.search(rf"{_upper}\s+{_num}\s+{_time}", q):
        filters["max_time_in_minutes"] = float(m.group(1))
    if m := re.search(rf"{_lower}\s+{_num}\s+{_time}", q):
        filters["min_time_in_minutes"] = float(m.group(1))
    if m := re.search(rf"between\s+{_num}\s+and\s+{_num}\s+{_time}", q):
        filters["min_time_in_minutes"] = float(m.group(1))
        filters["max_time_in_minutes"] = float(m.group(2))
    if m := re.search(rf"{_num}\s+{_time}\s+or\s+less", q):
        filters["max_time_in_minutes"] = float(m.group(1))
    if m := re.search(rf"{_num}\s+{_time}\s+or\s+more", q):
        filters["min_time_in_minutes"] = float(m.group(1))

    # ---- Calories ----
    if re.search(r"low[\s-]?calorie", q):
        filters.setdefault("max_calories", 200.0)
    if m := re.search(rf"{_upper}\s+{_num}\s+{_cal}", q):
        filters["max_calories"] = float(m.group(1))
    if m := re.search(rf"{_lower}\s+{_num}\s+{_cal}", q):
        filters["min_calories"] = float(m.group(1))
    if m := re.search(rf"between\s+{_num}\s+and\s+{_num}\s+{_cal}", q):
        filters["min_calories"] = float(m.group(1))
        filters["max_calories"] = float(m.group(2))
    if m := re.search(rf"{_num}\s+{_cal}\s+or\s+less", q):
        filters["max_calories"] = float(m.group(1))
    if m := re.search(rf"{_num}\s+{_cal}\s+or\s+more", q):
        filters["min_calories"] = float(m.group(1))

    # ---- Ingredient count ----
    if m := re.search(rf"{_upper}\s+{_num}\s+{_ing}", q):
        filters["max_ingredient_count"] = float(m.group(1))
    if m := re.search(rf"{_lower}\s+{_num}\s+{_ing}", q):
        filters["min_ingredient_count"] = float(m.group(1))
    if m := re.search(rf"between\s+{_num}\s+and\s+{_num}\s+{_ing}", q):
        filters["min_ingredient_count"] = float(m.group(1))
        filters["max_ingredient_count"] = float(m.group(2))
    if m := re.search(rf"{_num}\s+{_ing}\s+or\s+(?:less|fewer)", q):
        filters["max_ingredient_count"] = float(m.group(1))
    if m := re.search(rf"{_num}\s+{_ing}\s+or\s+more", q):
        filters["min_ingredient_count"] = float(m.group(1))

    return filters


# ---------------------------------------------------------------------------
# Filtered retrieval
# ---------------------------------------------------------------------------
def retrieve_with_filters(
    query: str,
    collection: chromadb.Collection,
    embed_model: SentenceTransformer,
    *,
    cuisine: str | None = None,
    min_time_in_minutes: float | None = None,
    max_time_in_minutes: float | None = None,
    min_calories: float | None = None,
    max_calories: float | None = None,
    min_ingredient_count: float | None = None,
    max_ingredient_count: float | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k semantically similar recipes with optional filters on
    cuisine, cooking time, calorie count, and ingredient count.

    All numeric filters and cuisine are auto-detected from the query text.
    Explicit keyword arguments override auto-detected values.

    Args:
        query                : Natural language search query.
        collection           : ChromaDB collection to search.
        embed_model          : SentenceTransformer used to encode the query.
        cuisine              : Cuisine override (auto-detected if omitted).
        min_time_in_minutes  : Lower bound on cook time in minutes (inclusive).
        max_time_in_minutes  : Upper bound on cook time in minutes (inclusive).
        min_calories         : Lower bound on calories per serving (inclusive).
        max_calories         : Upper bound on calories per serving (inclusive).
        min_ingredient_count : Lower bound on number of ingredients (inclusive).
        max_ingredient_count : Upper bound on number of ingredients (inclusive).
        top_k                : Number of results to return.

    Returns:
        List of dicts with keys: document, metadata, distance.
        Empty list if no recipes match the filters.
    """
    # --- Auto-detect numeric filters from query, then let explicit args override ---
    auto = _parse_filters_from_query(query)
    min_time_in_minutes  = min_time_in_minutes  if min_time_in_minutes  is not None else auto.get("min_time_in_minutes")
    max_time_in_minutes  = max_time_in_minutes  if max_time_in_minutes  is not None else auto.get("max_time_in_minutes")
    min_calories         = min_calories         if min_calories         is not None else auto.get("min_calories")
    max_calories         = max_calories         if max_calories         is not None else auto.get("max_calories")
    min_ingredient_count = min_ingredient_count if min_ingredient_count is not None else auto.get("min_ingredient_count")
    max_ingredient_count = max_ingredient_count if max_ingredient_count is not None else auto.get("max_ingredient_count")

    where_clauses: list[dict] = []

    # --- Cuisine ---
    if cuisine is not None:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        matched = list({m["cuisine"] for m in all_meta if cuisine.lower() in m["cuisine"].lower()})
        if not matched:
            print(f"[retrieve] No recipes found for cuisine='{cuisine}'.")
            return []
        where_clauses.append({"cuisine": {"$in": matched}})
        print(f"[retrieve] Cuisine (explicit): {matched}")
    else:
        auto_cuisines = _detect_cuisine_from_query(query, collection)
        if auto_cuisines:
            where_clauses.append({"cuisine": {"$in": auto_cuisines}})
            print(f"[retrieve] Cuisine (auto-detected): {auto_cuisines}")
        else:
            print("[retrieve] No cuisine detected — searching all cuisines.")

    # --- Cooking time ---
    if min_time_in_minutes is not None:
        where_clauses.append({"total_time_in_minutes": {"$gte": float(min_time_in_minutes)}})
    if max_time_in_minutes is not None:
        where_clauses.append({"total_time_in_minutes": {"$lte": float(max_time_in_minutes)}})

    # --- Calories ---
    if min_calories is not None:
        where_clauses.append({"calories": {"$gte": float(min_calories)}})
    if max_calories is not None:
        where_clauses.append({"calories": {"$lte": float(max_calories)}})

    # --- Ingredient count ---
    if min_ingredient_count is not None:
        where_clauses.append({"ingredient_count": {"$gte": float(min_ingredient_count)}})
    if max_ingredient_count is not None:
        where_clauses.append({"ingredient_count": {"$lte": float(max_ingredient_count)}})

    # Log all active filters
    active_numeric = {k: v for k, v in {
        "min_time": min_time_in_minutes, "max_time": max_time_in_minutes,
        "min_cal":  min_calories,        "max_cal":  max_calories,
        "min_ing":  min_ingredient_count,"max_ing":  max_ingredient_count,
    }.items() if v is not None}
    if active_numeric:
        print(f"[retrieve] Numeric filters: {active_numeric}")

    where = (
        {"$and": where_clauses} if len(where_clauses) > 1
        else where_clauses[0]   if len(where_clauses) == 1
        else None
    )

    query_embedding = embed_model.encode([query]).tolist()
    query_kwargs: dict = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    return [
        {"document": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ---------------------------------------------------------------------------
# Local generation with Phi-4 via Ollama
# ---------------------------------------------------------------------------
def generate_with_phi4(
    query: str,
    retrieved_docs: list[dict],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_tokens: int = 512,
) -> str:
    """
    Generate a grounded answer using the local Phi-4 model via Ollama.

    Args:
        query          : The user's question.
        retrieved_docs : Output of retrieve_with_filters().
        model          : Ollama model tag (default: phi4-mini:3.8b).
        base_url       : Ollama server URL (default: http://localhost:11434).
        max_tokens     : Maximum tokens in the generated response.

    Returns:
        The model's answer as a string.

    Raises:
        RuntimeError if Ollama returns a non-200 status.
    """
    if not retrieved_docs:
        return "No matching recipes were found for the given filters and query."

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
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from rag_pipeline import load_pipeline, list_collections, COLLECTION_NAME

    parser = argparse.ArgumentParser(
        description="Query any ChromaDB collection via the RAG pipeline.\n"
                    "Defaults to the Indian recipes collection.\n"
                    "Pass --collection to query a different dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("query",           nargs="?",  default=None,
                        help="Search query (prompted interactively if omitted).")
    parser.add_argument("--collection",    type=str,   default=COLLECTION_NAME,
                        help=f"ChromaDB collection to query (default: {COLLECTION_NAME}).\n"
                             "Run with --list to see all available collections.")
    parser.add_argument("--list",          action="store_true",
                        help="List all available collections and exit.")
    parser.add_argument("--cuisine",       type=str,   default=None,
                        help="Cuisine filter (recipes collection only).")
    parser.add_argument("--min-time",      type=float, default=None, metavar="MINS")
    parser.add_argument("--max-time",      type=float, default=None, metavar="MINS")
    parser.add_argument("--min-calories",  type=float, default=None, metavar="KCAL")
    parser.add_argument("--max-calories",  type=float, default=None, metavar="KCAL")
    parser.add_argument("--min-ing",       type=float, default=None, metavar="N")
    parser.add_argument("--max-ing",       type=float, default=None, metavar="N")
    parser.add_argument("--top-k",         type=int,   default=5,
                        help="Number of results to retrieve (default: 5).")
    args = parser.parse_args()

    if args.list:
        cols = list_collections()
        print("Collections in chroma_db/:")
        for name in cols:
            print(f"  • {name}")
        raise SystemExit(0)

    query = args.query or input("Enter your query: ").strip()
    if not query:
        parser.error("Query cannot be empty.")

    print(f"\nLoading collection '{args.collection}' …")
    embed_model, collection = load_pipeline(collection_name=args.collection)

    print(f"\nQuery : {query}")

    detected = _parse_filters_from_query(query)
    if detected:
        print("Parsed from query:", detected)
    print("-" * 60)

    docs = retrieve_with_filters(
        query, collection, embed_model,
        cuisine=args.cuisine,
        min_time_in_minutes=args.min_time,
        max_time_in_minutes=args.max_time,
        min_calories=args.min_calories,
        max_calories=args.max_calories,
        min_ingredient_count=args.min_ing,
        max_ingredient_count=args.max_ing,
        top_k=args.top_k,
    )

    if not docs:
        print("\nNo results found.")
    else:
        print("\n" + generate_with_phi4(query, docs))
