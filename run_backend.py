"""
Quick script to run the backend API.
"""
import uvicorn
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    print("Starting DTI Prediction API...")
    print("-" * 60)
    print("API will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("-" * 60)
    
    from backend.main import app
    uvicorn.run(app, host="0.0.0.0", port=8000)
