import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")

try:
    from data_collection import load_raw_csv, get_city_data
    print("✓ data_collection imported")
except ImportError as e:
    print(f"✗ data_collection failed: {e}")

try:
    from data_cleaning import remove_anomalies, create_daily_series
    print("✓ data_cleaning imported")
except ImportError as e:
    print(f"✗ data_cleaning failed: {e}")

try:
    from analysis import create_analysis_summary
    print("✓ analysis imported")
except ImportError as e:
    print(f"✗ analysis failed: {e}")

try:
    from forecasting import compare_models
    print("✓ forecasting imported")
except ImportError as e:
    print(f"✗ forecasting failed: {e}")

try:
    from visualization import plot_daily_duration_by_city
    print("✓ visualization imported")
except ImportError as e:
    print(f"✗ visualization failed: {e}")

print("\nTesting data loading...")

data_path = Path(__file__).parent.parent / "data" / "raw.csv"
if data_path.exists():
    df = load_raw_csv(data_path)
    print(f"✓ Data loaded: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
else:
    print(f"✗ Data file not found: {data_path}")

print("\nAll imports successful!")
