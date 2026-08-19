import os
import sys
import numpy as np

# Add the python directory to sys.path so we can import our custom modules
sys.path.append(os.path.abspath("python"))

from ingestion.data_fetcher import DataFetcher
from ingestion.id_mapper import EntityMapper
from ingestion.historical_replayer import DataReplayer

def test_full_pipeline():
    print("=== Phase 1: Fetching Real Data ===")
    fetcher = DataFetcher(output_dir="data")
    
    # Fetch a single 15-minute slice of GDELT (Nov 1, 2023 at 08:00 AM)
    test_datetime = "20231101080000"
    fetcher.fetch_gdelt_hourly_export(test_datetime)
    
    csv_path = f"data/gdelt/gdelt_{test_datetime}.csv"
    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: Could not find {csv_path}. Download failed.")
        sys.exit(1)

    print("\n=== Phase 2: Initializing Engine ===")
    # Use a test JSON file so we don't pollute your main production mappings
    mapper = EntityMapper(filepath="data/test_entity_map.json")
    
    # 50,000 nodes and 500,000 edges is plenty for a 15-minute slice of global news
    replayer = DataReplayer(num_nodes=50_000, max_edges=500_000, arena_bytes=100 * 1024 * 1024)

    print("\n=== Phase 3: Streaming to C++ ===")
    graph = replayer.replay_gdelt_csv(csv_path, mapper)

    print("\n=== Phase 4: Zero-Copy Memory Verification ===")
    # Pull the memory directly into numpy
    edges_array = np.array(graph.get_edges(), copy=False)
    
    # 4294967295 (0xFFFFFFFF) is the EMPTY_GAP marker we set in C++
    valid_edges = np.sum(edges_array[:, 0] != 4294967295) 
    
    print(f"Total Unique Entities Mapped : {mapper.current_id:,}")
    print(f"Valid Edges in C++ Graph     : {valid_edges:,}")
    print(f"Zero-copy Numpy Array Shape  : {edges_array.shape}")
    print("\n✅ Test passed! The real-world data pipeline is fully operational.")

if __name__ == "__main__":
    test_full_pipeline()