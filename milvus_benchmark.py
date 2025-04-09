import numpy as np
import time
from tqdm import tqdm
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def benchmark_milvus(vectors, queries, top_k):
    collection_name = "benchmark_vectors"
    dim = vectors.shape[1]

    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port="19530")

    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Benchmark vector collection")
    collection = Collection(name=collection_name, schema=schema)

    # Insert vectors
    print("Uploading to Milvus...")
    ids = list(range(len(vectors)))
    collection.insert([ids, vectors.tolist()])
    collection.flush()

    # Create index
    print("Creating index...")
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name="vector", index_params=index_params)

    # Load collection into memory
    collection.load()

    # Query
    latencies = []
    top1_scores = []

    for query in tqdm(queries, desc="Querying Milvus"):
        start = time.time()
        results = collection.search(
            data=[query.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["id"]
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        if results[0]:
            top_vec_id = results[0][0].id
            top_vec = vectors[top_vec_id]
            score = cosine_similarity(query, top_vec)
            top1_scores.append(score)
        else:
            top1_scores.append(0.0)

    avg_latency = np.mean(latencies)
    avg_score = np.mean(top1_scores)

    print("\\n===== Milvus Benchmark =====")
    print(f"Avg Query Latency: {avg_latency:.2f} ms")
    print(f"Avg Top-1 Cosine Similarity: {avg_score:.4f}")

    return {
        "Database": "Milvus",
        "Avg Query Latency (ms)": round(avg_latency, 2),
        "Avg Top-1 Cosine Similarity": round(avg_score, 4)
    }

