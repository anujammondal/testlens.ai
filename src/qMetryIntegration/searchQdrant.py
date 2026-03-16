"""
Search QMetry Test Cases in Qdrant by Semantic Similarity

Usage:
    python src/qMetryIntegration/searchQdrant.py "your search query"
    python src/qMetryIntegration/searchQdrant.py "user login" --limit 10
    python src/qMetryIntegration/searchQdrant.py --interactive

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - Collection created: python src/qMetryIntegration/uploadToQdrant.py
"""

import argparse
import sys
from typing import List, Dict, Any

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("❌ qdrant-client not installed. Run:")
    print("   pip install qdrant-client")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ sentence-transformers not installed. Run:")
    print("   pip install sentence-transformers")
    sys.exit(1)


# ---------------- CONFIG ----------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "qmetry_testcases"
# Must match the model used in createSemanticEmbeddings.py
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# Load model once
print(f"🔄 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("✅ Model loaded")


def connect_to_qdrant() -> QdrantClient:
    """Connect to Qdrant server."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("\n   Make sure Qdrant is running:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)


def search(
    client: QdrantClient,
    query: str,
    limit: int = 5,
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search for similar test cases using text query.
    
    Args:
        client: QdrantClient instance
        query: Text to search for
        limit: Number of results to return
        score_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of results with scores and metadata
    """
    # Generate embedding
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    
    # Search
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit
    ).points
    
    # Format results
    formatted = []
    for r in results:
        if r.score >= score_threshold:
            formatted.append({
                "score": r.score,
                "key": r.payload.get("key", ""),
                "summary": r.payload.get("summary", ""),
                "normalized_summary": r.payload.get("normalized_summary", ""),
                "priority": r.payload.get("priority", ""),
                "qmetry_id": r.payload.get("qmetry_id", "")
            })
    
    return formatted


def print_results(results: List[Dict[str, Any]], query: str) -> None:
    """Pretty print search results."""
    print(f"\n{'='*70}")
    print(f"🔎 Query: \"{query}\"")
    print(f"{'='*70}")
    
    if not results:
        print("\n   No results found.")
        return
    
    for i, r in enumerate(results, 1):
        score_bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        print(f"\n{i}. [{r['key']}] Score: {r['score']:.4f} |{score_bar}|")
        print(f"   Priority: {r['priority']}")
        print(f"   Summary: {r['summary']}")
        if r['normalized_summary'] and r['normalized_summary'] != r['summary']:
            print(f"   Normalized: {r['normalized_summary'][:100]}...")


def interactive_mode(client: QdrantClient, limit: int = 5) -> None:
    """Run interactive search mode."""
    print("\n" + "="*70)
    print("🔍 Interactive Search Mode")
    print("   Type your query and press Enter. Type 'quit' or 'q' to exit.")
    print("   Type 'limit N' to change number of results (e.g., 'limit 10')")
    print("="*70)
    
    current_limit = limit
    
    while True:
        try:
            query = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ('quit', 'q', 'exit'):
            print("\n👋 Goodbye!")
            break
        
        # Check for limit command
        if query.lower().startswith('limit '):
            try:
                current_limit = int(query.split()[1])
                print(f"   ✅ Limit set to {current_limit}")
            except (ValueError, IndexError):
                print("   ❌ Invalid limit. Usage: limit 10")
            continue
        
        # Perform search
        results = search(client, query, limit=current_limit)
        print_results(results, query)


def main():
    parser = argparse.ArgumentParser(
        description="Search QMetry test cases by semantic similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python searchQdrant.py "user authentication"
    python searchQdrant.py "video playback" --limit 10
    python searchQdrant.py "menu navigation" --threshold 0.5
    python searchQdrant.py --interactive
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query text"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.0,
        help="Minimum similarity score 0-1 (default: 0.0)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Connect to Qdrant
    print(f"\n🔌 Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = connect_to_qdrant()
    print("✅ Connected")
    
    # Interactive mode
    if args.interactive:
        interactive_mode(client, args.limit)
        return
    
    # Single query mode
    if not args.query:
        parser.print_help()
        print("\n❌ Please provide a search query or use --interactive mode")
        sys.exit(1)
    
    results = search(
        client,
        args.query,
        limit=args.limit,
        score_threshold=args.threshold
    )
    
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print_results(results, args.query)
        print(f"\n✅ Found {len(results)} results")


if __name__ == "__main__":
    main()
