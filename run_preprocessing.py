"""
Quick script to run data preprocessing.
"""
from data.preprocess import preprocess_bindingdb

if __name__ == '__main__':
    tsv_path = 'BindingDB_All_202511_tsv/BindingDB_All.tsv'
    print("Starting BindingDB preprocessing...")
    print(f"Input file: {tsv_path}")
    print("-" * 60)
    
    preprocess_bindingdb(tsv_path)
    
    print("-" * 60)
    print("Preprocessing complete! Processed data saved to data/processed/")
    print("\nNext steps:")
    print("1. Train the model: python training/train.py")
    print("2. Start the backend: python backend/main.py")
    print("3. Start the frontend: cd frontend && npm run dev")
