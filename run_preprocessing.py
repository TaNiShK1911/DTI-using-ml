"""
Quick script to run data preprocessing.
"""
from data.preprocess import preprocess_bindingdb

if __name__ == '__main__':
    tsv_path = 'BindingDB_All_202511_tsv/BindingDB_All.tsv'
    print("Starting BindingDB preprocessing...")
    print(f"Input file: {tsv_path}")
    print("-" * 60)
    print("Reading first 50,000 rows for prototype")
    print("Will filter to ~10,000 valid samples after processing")
    print("-" * 60)
    
    # Use direct reading (not chunks) with row limit for simplicity
    preprocess_bindingdb(tsv_path, use_chunks=False, max_rows=50_000, prototype_limit=10_000)
    
    print("-" * 60)
    print("Preprocessing complete! Processed data saved to data/processed/")
    print("\nNext steps:")
    print("1. Train the model: python run_training.py")
    print("2. Start the backend: python run_backend.py")
    print("3. Start the frontend: cd frontend && npm run dev")
