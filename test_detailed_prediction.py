"""
Test script for detailed prediction functionality.
"""
import requests
import json

def test_detailed_prediction():
    """Test the enhanced prediction endpoint."""
    
    # Test data
    test_data = {
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "sequence": "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQGSGQPTTPTGQEKPSPHSTLPQPKTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL"
    }
    
    try:
        # Make request to backend
        response = requests.post("http://localhost:8000/predict", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("🎉 Prediction successful!")
            print(f"Affinity: {result['affinity']:.3f}")
            print(f"Uncertainty: {result['uncertainty']:.3f}")
            print(f"Confidence: {result['confidence_level']}")
            
            # Print detailed explanation
            if 'detailed_explanation' in result:
                print("\n📊 Detailed Analysis:")
                
                # Affinity calculation
                affinity_calc = result['detailed_explanation']['affinity_calculation']
                print(f"\n🎯 Affinity: {affinity_calc['interpretation']}")
                print(f"   Binding Strength: {affinity_calc['binding_strength']}")
                
                # Uncertainty breakdown
                uncertainty = result['detailed_explanation']['uncertainty_breakdown']
                print(f"\n📈 Uncertainty Breakdown:")
                print(f"   Epistemic: {uncertainty['epistemic_uncertainty']:.4f}")
                print(f"   Aleatoric: {uncertainty['aleatoric_uncertainty']:.4f}")
                print(f"   Total: {uncertainty['total_uncertainty']:.4f}")
                print(f"   Confidence: {uncertainty['confidence_description']}")
                
                # Evidential parameters
                params = result['detailed_explanation']['evidential_parameters']
                print(f"\n🔬 Evidential Parameters:")
                print(f"   γ (gamma): {params['gamma']:.4f}")
                print(f"   ν (nu): {params['nu']:.4f}")
                print(f"   α (alpha): {params['alpha']:.4f}")
                print(f"   β (beta): {params['beta']:.4f}")
                
                # Attention analysis
                attention = result['detailed_explanation']['attention_analysis']
                print(f"\n🔍 Attention Analysis:")
                print(f"   Shape: {attention['attention_shape']}")
                print(f"   Range: {attention['min_attention']:.4f} - {attention['max_attention']:.4f}")
                
                # Input analysis
                input_analysis = result['detailed_explanation']['input_analysis']
                print(f"\n📝 Input Analysis:")
                print(f"   SMILES length: {input_analysis['smiles_length']}")
                print(f"   Sequence length: {input_analysis['sequence_length']}")
                
                if input_analysis['smiles_complexity']['num_atoms'] != "N/A":
                    complexity = input_analysis['smiles_complexity']
                    print(f"   Molecule: {complexity['num_atoms']} atoms, {complexity['num_bonds']} bonds, {complexity['num_rings']} rings")
                    print(f"   Molecular weight: {complexity['molecular_weight']:.2f} Da")
                
                composition = input_analysis['sequence_composition']
                print(f"   Protein: {composition['unique_amino_acids']} unique AAs, most common: {composition['most_common'][0]}")
            
            print(f"\n✅ Test completed successfully!")
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("Testing detailed prediction functionality...")
    print("Make sure the backend is running: python run_backend.py")
    print("-" * 60)
    test_detailed_prediction()