import numpy as np
import time
from tqdm import tqdm
import weaviate

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def benchmark_weaviate(vectors, queries, top_k):

    client = weaviate.Client("http://localhost:8080")

    class_name = "BenchmarkVector"

    # Define schema
    class_obj = {
        "class": class_name,
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [{"name": "text", "dataType": ["text"]}],
    }

    if client.schema.contains({"classes": [class_obj]}):
        client.schema.delete_class(class_name)
    client.schema.create_class(class_obj)


    # Upload vectors
    for i, vec in enumerate(tqdm(vectors, desc="Uploading to Weaviate")):
        client.data_object.create(
            data_object={"text": f"vec-{i}"},
            class_name=class_name,
            vector=vec.tolist()
        )

    # Query and measure performance
    latencies = []
    top1_scores = []

    for query in tqdm(queries, desc="Querying Weaviate"):
        start = time.time()
        result = client.query\
            .get(class_name, ["text", "_additional { vector }"])\
            .with_near_vector({"vector": query.tolist()})\
            .with_limit(top_k)\
            .do()
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        try:
            top_vec = result["data"]["Get"][class_name][0]["_additional"]["vector"]
            score = cosine_similarity(query, np.array(top_vec))
            top1_scores.append(score)
        except (KeyError, IndexError):
            top1_scores.append(0.0)

    avg_latency = np.mean(latencies)
    avg_score = np.mean(top1_scores)

    print("\\n===== Weaviate Benchmark =====")
    print(f"Avg Query Latency: {avg_latency:.2f} ms")
    print(f"Avg Top-1 Cosine Similarity: {avg_score:.4f}")

    return {
        "Database": "Weaviate",
        "Avg Query Latency (ms)": round(avg_latency, 2),
        "Avg Top-1 Cosine Similarity": round(avg_score, 4)
    }

