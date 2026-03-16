"""
Upload QMetry Test Case Embeddings to Qdrant Vector Database

This script:
1. Loads embeddings from qmetry_testcases_embeddings.json
2. Creates a Qdrant collection with appropriate vector configuration
3. Uploads all test case embeddings with metadata
4. Provides verification and search functionality

Prerequisites:
    - Qdrant running locally: docker run -p 6333:6333 qdrant/qdrant
    - Install client: pip install qdrant-client
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue
    )
except ImportError:
    print("❌ qdrant-client not installed. Run:")
    print("   pip install qdrant-client")
    sys.exit(1)


# ---------------- CONFIG ----------------
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "qmetry_testcases"
EMBEDDINGS_FILE = "qmetry_testcases_embeddings.json"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_embeddings(file_path: Path) -> Dict[str, Any]:
    """
    Load embeddings from JSON file.
    
    Returns:
        Dictionary containing model info and test cases with embeddings
    """
    print(f"📂 Loading embeddings from: {file_path}")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print("   Run createSemanticEmbeddings.py first to generate embeddings.")
        sys.exit(1)
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ Loaded {data['total_testcases']} test cases")
    print(f"   Model: {data['model']}")
    print(f"   Embedding dimension: {data['embedding_dimension']}")
    
    return data


def connect_to_qdrant(host: str = QDRANT_HOST, port: int = QDRANT_PORT) -> QdrantClient:
    """
    Connect to Qdrant server.
    
    Returns:
        QdrantClient instance
    """
    print(f"🔌 Connecting to Qdrant at {host}:{port}...")
    
    try:
        client = QdrantClient(host=host, port=port)
        # Test connection
        client.get_collections()
        print("✅ Connected to Qdrant")
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        print("\n   Make sure Qdrant is running:")
        print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        sys.exit(1)


def create_collection(
    client: QdrantClient,
    collection_name: str,
    embedding_dimension: int,
    recreate: bool = True
) -> None:
    """
    Create a Qdrant collection for storing embeddings.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        embedding_dimension: Size of embedding vectors
        recreate: If True, delete existing collection and create new one
    """
    print(f"\n📦 Setting up collection: {collection_name}")
    
    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if exists and recreate:
        print(f"   Deleting existing collection...")
        client.delete_collection(collection_name)
    elif exists and not recreate:
        print(f"   Collection already exists, skipping creation")
        return
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dimension,
            distance=Distance.COSINE
        )
    )
    print(f"✅ Collection '{collection_name}' created")
    print(f"   Vector size: {embedding_dimension}")
    print(f"   Distance metric: Cosine")


def prepare_points(
    testcases: List[Dict[str, Any]],
    project_id: str = "",
    parent_folder: Dict[str, Any] = None
) -> List[PointStruct]:
    """
    Convert test cases to Qdrant points.
    
    Args:
        testcases: List of test case dictionaries with embeddings
        project_id: Project ID from the source data
        parent_folder: Parent folder info (id and name)
        
    Returns:
        List of PointStruct objects ready for upload
    """
    points = []
    parent_folder = parent_folder or {}
    
    for idx, tc in enumerate(testcases):
        # Extract step details as a single string for payload
        steps_text = ""
        if "stepDetails" in tc:
            steps_text = " | ".join([
                f"{k}: {v}" for k, v in tc["stepDetails"].items() if v
            ])
        
        # Extract folder info
        folder = tc.get("folder", {})
        
        point = PointStruct(
            id=idx,
            vector=tc["embedding"],
            payload={
                "qmetry_id": tc.get("id", ""),
                "key": tc.get("key", ""),
                "project_id": project_id,
                "parent_folder_id": parent_folder.get("id", ""),
                "parent_folder_name": parent_folder.get("name", ""),
                "folder_id": folder.get("id", ""),
                "folder_name": folder.get("name", ""),
                "summary": tc.get("summary", ""),
                "normalized_summary": tc.get("normalized_summary", ""),
                "priority": tc.get("priority", ""),
                "steps_count": tc.get("stepsCount", 0),
                "steps_text": steps_text
            }
        )
        points.append(point)
    
    return points


def upload_points(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 100
) -> None:
    """
    Upload points to Qdrant collection in batches.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        points: List of PointStruct objects
        batch_size: Number of points to upload per batch
    """
    print(f"\n📤 Uploading {len(points)} points to '{collection_name}'...")
    
    # Upload in batches
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"   Uploaded batch {i//batch_size + 1}: {len(batch)} points")
    
    print(f"✅ All points uploaded successfully")


def verify_collection(client: QdrantClient, collection_name: str) -> None:
    """
    Verify the collection was created and populated correctly.
    """
    print(f"\n🔍 Verifying collection...")
    
    info = client.get_collection(collection_name)
    print(f"   Points count: {info.points_count}")
    print(f"   Status: {info.status}")


def test_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 5
) -> None:
    """
    Test search functionality with a sample query.
    """
    print(f"\n🔎 Testing search (top {limit} results)...")
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit
    ).points
    
    for i, result in enumerate(results, 1):
        print(f"\n   {i}. Score: {result.score:.4f}")
        print(f"      Key: {result.payload.get('key', 'N/A')}")
        print(f"      Summary: {result.payload.get('summary', 'N/A')[:80]}...")


def search_by_text(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for similar test cases using text query.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        query_text: Text to search for
        model_name: Sentence transformer model name
        limit: Number of results to return
        
    Returns:
        List of search results with scores and payloads
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence-transformers not installed for text search")
        return []
    
    print(f"\n🔎 Searching for: '{query_text[:50]}...'")
    
    # Generate embedding for query
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()
    
    # Search
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit
    ).points
    
    # Format results
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append({
            "rank": i,
            "score": result.score,
            "key": result.payload.get("key", ""),
            "summary": result.payload.get("summary", ""),
            "priority": result.payload.get("priority", "")
        })
        print(f"\n   {i}. Score: {result.score:.4f}")
        print(f"      Key: {result.payload.get('key', 'N/A')}")
        print(f"      Summary: {result.payload.get('summary', 'N/A')}")
    
    return formatted_results


def main():
    """Main function to upload embeddings to Qdrant."""
    print("=" * 60)
    print("QMetry Test Cases → Qdrant Upload")
    print("=" * 60)
    
    # Get file paths
    project_root = get_project_root()
    embeddings_path = project_root / EMBEDDINGS_FILE
    
    # Load embeddings
    data = load_embeddings(embeddings_path)
    
    # Connect to Qdrant
    client = connect_to_qdrant()
    
    # Create collection
    create_collection(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_dimension=data["embedding_dimension"],
        recreate=True
    )
    
    # Prepare and upload points
    points = prepare_points(
        testcases=data["testcases"],
        project_id=data.get("projectId", ""),
        parent_folder=data.get("parentFolder", {})
    )
    upload_points(client, COLLECTION_NAME, points)
    
    # Verify
    verify_collection(client, COLLECTION_NAME)
    
    # Test search with first embedding
    if data["testcases"]:
        first_embedding = data["testcases"][0]["embedding"]
        test_search(client, COLLECTION_NAME, first_embedding)
    
    print("\n" + "=" * 60)
    print("✅ Upload complete!")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")
    print("=" * 60)


if __name__ == "__main__":
    main()
