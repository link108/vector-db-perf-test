import numpy as np
import time
from tqdm import tqdm
from pymongo import MongoClient
from pymongo.errors import OperationFailure

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def benchmark_mongo(vectors, queries, top_k):
    client = MongoClient("mongodb://localhost:27017")
    db = client["benchmark"]
    collection = db["vectors"]
    
    # Drop and recreate collection
    collection.drop()

    print("Uploading to MongoDB...")
    docs = [{"_id": i, "vector": vec.tolist(), "text": f"vec-{i}"} for i, vec in enumerate(vectors)]
    collection.insert_many(docs)

    print("Creating vector index...")
    try:
        collection.create_index(
            [("vector", "knnVector")],
            name="vector_index",
            default_language="none",
            knnVectorOptions={
                "dimensions": vectors.shape[1],
                "similarity": "cosine"
            }
        )
    except OperationFailure as e:
        print("Index creation failed — are vector search features enabled in your MongoDB?")
        return {
            "Database": "MongoDB",
            "Avg Query Latency (ms)": -1,
            "Avg Top-1 Cosine Similarity": 0.0
        }

    latencies = []
    top1_scores = []

    for query in tqdm(queries, desc="Querying MongoDB"):
        start = time.time()
        try:
            results = list(collection.aggregate([
                {
                    "$vectorSearch": {
                        "queryVector": query.tolist(),
                        "path": "vector",
                        "numCandidates": 100,
                        "limit": top_k,
                        "index": "vector_index"
                    }
                }
            ]))
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            if results:
                top_vec = np.array(results[0]["vector"])
                score = cosine_similarity(query, top_vec)
                top1_scores.append(score)
            else:
                top1_scores.append(0.0)
        except Exception as e:
            print("Query failed:", e)
            top1_scores.append(0.0)
            latencies.append(0)

    avg_latency = np.mean(latencies)
    avg_score = np.mean(top1_scores)

    print("\\n===== MongoDB Benchmark =====")
    print(f"Avg Query Latency: {avg_latency:.2f} ms")
    print(f"Avg Top-1 Cosine Similarity: {avg_score:.4f}")

    client.close()
    return {
        "Database": "MongoDB",
        "Avg Query Latency (ms)": round(avg_latency, 2),
        "Avg Top-1 Cosine Similarity": round(avg_score, 4)
    }

