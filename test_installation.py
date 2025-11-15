"""
Test script to verify the installation and basic functionality.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('fastapi', 'FastAPI'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('rdkit', 'RDKit'),
        ('scipy', 'SciPy'),
        ('pydantic', 'Pydantic'),
        ('uvicorn', 'Uvicorn'),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - not installed")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_model_components():
    """Test if model components can be imported."""
    print("\nTesting model components...")
    
    try:
        from models.encoders import DrugEncoder, ProteinEncoder
        print("  ✅ Encoders")
    except Exception as e:
        print(f"  ❌ Encoders: {e}")
        return False
    
    try:
        from models.attention import CoAttentionModule
        print("  ✅ Co-attention")
    except Exception as e:
        print(f"  ❌ Co-attention: {e}")
        return False
    
    try:
        from models.evidential import EvidentialRegressionHead, EvidentialLoss
        print("  ✅ Evidential regression")
    except Exception as e:
        print(f"  ❌ Evidential regression: {e}")
        return False
    
    try:
        from models.dti_model import DTIModel
        print("  ✅ DTI Model")
    except Exception as e:
        print(f"  ❌ DTI Model: {e}")
        return False
    
    return True


def test_data_components():
    """Test if data components can be imported."""
    print("\nTesting data components...")
    
    try:
        from data.preprocess import BindingDBParser, AffinityProcessor, DataSplitter
        print("  ✅ Preprocessing")
    except Exception as e:
        print(f"  ❌ Preprocessing: {e}")
        return False
    
    try:
        from data.dataset import DTIDataset
        print("  ✅ Dataset")
    except Exception as e:
        print(f"  ❌ Dataset: {e}")
        return False
    
    return True


def test_backend():
    """Test if backend can be imported."""
    print("\nTesting backend...")
    
    try:
        from backend.main import app, ModelService
        print("  ✅ Backend API")
    except Exception as e:
        print(f"  ❌ Backend API: {e}")
        return False
    
    return True


def test_training():
    """Test if training components can be imported."""
    print("\nTesting training components...")
    
    try:
        from training.train import Trainer
        print("  ✅ Training pipeline")
    except Exception as e:
        print(f"  ❌ Training pipeline: {e}")
        return False
    
    return True


def test_rdkit_functionality():
    """Test basic RDKit functionality."""
    print("\nTesting RDKit functionality...")
    
    try:
        from rdkit import Chem
        
        # Test SMILES parsing
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is not None:
            print(f"  ✅ SMILES parsing works")
            return True
        else:
            print(f"  ❌ SMILES parsing failed")
            return False
    except Exception as e:
        print(f"  ❌ RDKit test failed: {e}")
        return False


def test_torch_functionality():
    """Test basic PyTorch functionality."""
    print("\nTesting PyTorch functionality...")
    
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠️  CUDA not available (will use CPU)")
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.matmul(x, y)
        
        print(f"  ✅ Tensor operations work")
        return True
    except Exception as e:
        print(f"  ❌ PyTorch test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DTI Prediction Platform - Installation Test")
    print("=" * 60)
    
    results = []
    
    # Test imports
    success, failed = test_imports()
    results.append(success)
    
    if not success:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return
    
    # Test components
    results.append(test_model_components())
    results.append(test_data_components())
    results.append(test_backend())
    results.append(test_training())
    
    # Test functionality
    results.append(test_rdkit_functionality())
    results.append(test_torch_functionality())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed!")
        print("\nYour installation is ready to use.")
        print("\nNext steps:")
        print("1. Preprocess data: python run_preprocessing.py")
        print("2. Train model: python run_training.py")
        print("3. Start backend: python run_backend.py")
        print("4. Start frontend: cd frontend && npm run dev")
    else:
        print("⚠️  Some tests failed")
        print("Please check the errors above and fix them before proceeding")
    print("=" * 60)


if __name__ == "__main__":
    main()
