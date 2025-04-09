import numpy as np
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def benchmark_qdrant(vectors, queries, top_k):
    client = QdrantClient(host="localhost", port=6333)

    collection_name = "benchmark_vectors"
    vector_dim = vectors.shape[1]

    # Drop collection if it exists
    if collection_name in client.get_collections().collections:
        client.delete_collection(collection_name)

    # Create collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    # Upload vectors in batches
    batch_size = 1000
    print("Uploading to Qdrant...")
    for i in tqdm(range(0, len(vectors), batch_size)):
        batch_vectors = vectors[i:i+batch_size]
        points = [
            PointStruct(id=i + j, vector=vec.tolist(), payload={"text": f"vec-{i + j}"})
            for j, vec in enumerate(batch_vectors)
        ]
        client.upsert(collection_name=collection_name, points=points)

    # Query
    latencies = []
    top1_scores = []

    for query in tqdm(queries, desc="Querying Qdrant"):
        start = time.time()
        hits = client.search(
            collection_name=collection_name,
            query_vector=query.tolist(),
            limit=top_k,
            with_vectors=True  # this is valid
        )

        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if hits:
            top_vec = np.array(hits[0].vector)
            score = cosine_similarity(query, top_vec)
            top1_scores.append(score)
        else:
            top1_scores.append(0.0)

    avg_latency = np.mean(latencies)
    avg_score = np.mean(top1_scores)

    print("\\n===== Qdrant Benchmark =====")
    print(f"Avg Query Latency: {avg_latency:.2f} ms")
    print(f"Avg Top-1 Cosine Similarity: {avg_score:.4f}")

    return {
        "Database": "Qdrant",
        "Avg Query Latency (ms)": round(avg_latency, 2),
        "Avg Top-1 Cosine Similarity": round(avg_score, 4)
    }

