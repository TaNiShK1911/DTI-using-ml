"""
Quick script to run model training.
"""
from training.train import main

if __name__ == '__main__':
    print("Starting DTI model training...")
    print("-" * 60)
    
    main()
    
    print("-" * 60)
    print("Training complete! Model saved to checkpoints/best_model.pt")
    print("\nNext steps:")
    print("1. Start the backend: python backend/main.py")
    print("2. Start the frontend: cd frontend && npm run dev")
    print("3. Access the app at http://localhost:3000")
