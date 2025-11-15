"""
Setup script to verify environment and dependencies.
"""
import sys
import subprocess
import os


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_pip():
    """Check if pip is available."""
    print("\nChecking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except:
        print("❌ pip not found")
        return False


def install_requirements():
    """Install Python requirements."""
    print("\nInstalling Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                      check=True)
        print("✅ Python dependencies installed")
        return True
    except:
        print("❌ Failed to install Python dependencies")
        return False


def check_node():
    """Check if Node.js is available."""
    print("\nChecking Node.js...")
    try:
        result = subprocess.run(["node", "--version"], 
                              check=True, capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"✅ Node.js {version}")
        return True
    except:
        print("❌ Node.js not found (required for frontend)")
        print("   Install from: https://nodejs.org/")
        return False


def install_frontend_deps():
    """Install frontend dependencies."""
    print("\nInstalling frontend dependencies...")
    try:
        os.chdir("frontend")
        subprocess.run(["npm", "install"], check=True)
        os.chdir("..")
        print("✅ Frontend dependencies installed")
        return True
    except:
        print("❌ Failed to install frontend dependencies")
        os.chdir("..")
        return False


def check_data():
    """Check if BindingDB data exists."""
    print("\nChecking data...")
    data_path = "BindingDB_All_202511_tsv/BindingDB_All.tsv"
    if os.path.exists(data_path):
        print(f"✅ BindingDB data found: {data_path}")
        return True
    else:
        print(f"⚠️  BindingDB data not found: {data_path}")
        print("   The data file should be placed in the BindingDB_All_202511_tsv/ directory")
        return False


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    dirs = ["data/processed", "checkpoints", "logs"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directories created")


def main():
    """Main setup function."""
    print("=" * 60)
    print("DTI Prediction Platform - Setup")
    print("=" * 60)
    
    checks = []
    
    # Check Python
    checks.append(check_python_version())
    
    # Check pip
    checks.append(check_pip())
    
    # Install Python dependencies
    if checks[-1]:
        checks.append(install_requirements())
    
    # Check Node.js
    node_available = check_node()
    checks.append(node_available)
    
    # Install frontend dependencies
    if node_available:
        checks.append(install_frontend_deps())
    
    # Check data
    check_data()  # Not critical for setup
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ Setup complete!")
        print("\nNext steps:")
        print("1. Preprocess data: python run_preprocessing.py")
        print("2. Train model: python run_training.py")
        print("3. Start backend: python run_backend.py")
        print("4. Start frontend: cd frontend && npm run dev")
    else:
        print("⚠️  Setup completed with warnings")
        print("Please resolve the issues above before proceeding")
    print("=" * 60)


if __name__ == "__main__":
    main()
