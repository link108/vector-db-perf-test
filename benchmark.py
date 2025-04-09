import numpy as np
import os
import pandas as pd
import time
import json

from weaviate_benchmark import benchmark_weaviate
from qdrant_benchmark import benchmark_qdrant
from milvus_benchmark import benchmark_milvus
from mongo_benchmark import benchmark_mongo

VECTOR_DIM = 768
NUM_VECTORS = 10000
NUM_QUERIES = 100
TOP_K = 10
DATA_DIR = "data"
RESULTS_JSON = "benchmark_results.json"
RESULTS_CSV = "benchmark_results.csv"

os.makedirs(DATA_DIR, exist_ok=True)

def generate_or_load_data():
    vec_path = os.path.join(DATA_DIR, "test_vectors.npy")
    qry_path = os.path.join(DATA_DIR, "test_queries.npy")

    if not os.path.exists(vec_path) or not os.path.exists(qry_path):
        print("🚀 Generating random vectors and queries...")
        vectors = np.random.rand(NUM_VECTORS, VECTOR_DIM).astype(np.float32)
        queries = np.random.rand(NUM_QUERIES, VECTOR_DIM).astype(np.float32)
        np.save(vec_path, vectors)
        np.save(qry_path, queries)
    else:
        print("📂 Loading existing vectors and queries...")

    vectors = np.load(vec_path)
    queries = np.load(qry_path)
    return vectors, queries

def load_existing_results():
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON, "r") as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    vectors, queries = generate_or_load_data()

    print(f"📊 Starting benchmark for {NUM_VECTORS} vectors, {NUM_QUERIES} queries, top_k={TOP_K}")
    print("=" * 60)

    existing_results = load_existing_results()
    results = []

    benchmarks = [
        ("Weaviate", benchmark_weaviate),
        ("Qdrant", benchmark_qdrant),
        ("Milvus", benchmark_milvus),
        ("MongoDB", benchmark_mongo),
    ]

    total_start = time.time()

    for name, func in benchmarks:
        if name in existing_results:
            print(f"⏭️ Skipping {name} (already completed)")
            results.append(existing_results[name])
            continue

        print(f"\n🔍 Running benchmark for: {name}")
        try:
            start = time.time()
            result = func(vectors, queries, TOP_K)
            duration = time.time() - start
            print(f"✅ Finished {name} in {duration:.2f}s")
            existing_results[name] = result
            save_results(existing_results)
            results.append(result)
        except Exception as e:
            print(f"❌ {name} benchmark failed: {e}")

    total_duration = time.time() - total_start
    print("\n✅ All benchmarks complete.")
    print(f"⏱️ Total run time: {total_duration:.2f}s")

    df = pd.DataFrame(results)
    print("\n=== FINAL RESULTS ===")
    print(df.to_markdown(index=False))
    df.to_csv(RESULTS_CSV, index=False)

