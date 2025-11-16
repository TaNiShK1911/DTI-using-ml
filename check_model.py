import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Checking model checkpoint...")
print("=" * 60)

# Load checkpoint
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)

print("Checkpoint keys:", list(ckpt.keys()))
print("\nConfig:", ckpt.get('config'))
print("\nState dict has", len(ckpt['model_state_dict']), "parameters")

# Check last few parameters (prediction head should be trained)
print("\nLast 5 parameter keys (prediction head):")
for k in list(ckpt['model_state_dict'].keys())[-5:]:
    v = ckpt['model_state_dict'][k]
    print(f"  {k}")
    print(f"    Shape: {v.shape}, Mean: {v.float().mean():.6f}, Std: {v.float().std():.6f}")

# Check if prediction head parameters look trained
print("\n" + "=" * 60)
print("Checking if model is actually trained...")
print("=" * 60)

# Load the model and check
from models.dti_model import DTIModel

model = DTIModel(config=ckpt['config'])
model.load_state_dict(ckpt['model_state_dict'], strict=False)

# Check prediction head parameters
pred_head_params = list(model.prediction_head.parameters())
print(f"\nPrediction head has {len(pred_head_params)} parameter tensors")

for i, p in enumerate(pred_head_params[-3:]):
    print(f"\nParameter {i} (last layers):")
    print(f"  Shape: {p.shape}")
    print(f"  Mean: {p.data.mean():.6f}")
    print(f"  Std: {p.data.std():.6f}")
    print(f"  Min: {p.data.min():.6f}, Max: {p.data.max():.6f}")

print("\n" + "=" * 60)
print("Analysis:")
print("=" * 60)

# If std is very small or values are near initialization, model isn't trained
last_param = pred_head_params[-1]
if last_param.data.std() < 0.01:
    print("⚠ WARNING: Model appears UNTRAINED (very small std)")
    print("  The checkpoint may have been saved before any training occurred")
elif last_param.data.std() < 0.1:
    print("⚠ WARNING: Model appears PARTIALLY trained")
else:
    print("✓ Model appears trained (parameters have reasonable variance)")
