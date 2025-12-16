"""
Fusion Training Example

This script demonstrates how to train the Fusion system with different configurations:

Option 1: With pretrained SwinTExCo (recommended if you have it)
Option 2: Without pretrained SwinTExCo (random NonLocalNet initialization)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fusion_system import FusionSystem
from FusionNet.fusion_unet import SimpleFusionNet

# =============================================================================
# Configuration
# =============================================================================

# Paths
MEMFLOW_PATH = '../MemFlow'
SWINTEXCO_PATH = '../SwinSingle'
MEMFLOW_CKPT = 'checkpoints/memflow_best.pth'

# Choose your training strategy:
USE_PRETRAINED_SWINTEXCO = False  # ← Set to False for random NonLocalNet init

if USE_PRETRAINED_SWINTEXCO:
    SWINTEXCO_CKPT = 'checkpoints/swintexco_best/'  # Path to pretrained weights
else:
    SWINTEXCO_CKPT = None  # Random initialization with pretrained Swin backbone

# =============================================================================
# Create Fusion System
# =============================================================================

print("="*80)
print("Creating Fusion System")
print("="*80)

fusion_system = FusionSystem(
    memflow_path=MEMFLOW_PATH,
    swintexco_path=SWINTEXCO_PATH,
    memflow_ckpt=MEMFLOW_CKPT,
    swintexco_ckpt=SWINTEXCO_CKPT,  # None = random NonLocalNet + pretrained Swin
    fusion_net=SimpleFusionNet(),
    device='cuda'
)

print("\n" + "="*80)
print("Fusion System Created Successfully!")
print("="*80)

# =============================================================================
# Training Configuration
# =============================================================================

# Check trainable parameters
trainable_params = sum(p.numel() for p in fusion_system.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in fusion_system.swintexco.nonlocal_net.parameters())
total_params += sum(p.numel() for p in fusion_system.fusion_unet.parameters())

print(f"\nTrainable parameters: {trainable_params:,}")
print(f"  - SwinTExCo NonLocalNet: {sum(p.numel() for p in fusion_system.swintexco.nonlocal_net.parameters()):,}")
print(f"  - Fusion UNet: {sum(p.numel() for p in fusion_system.fusion_unet.parameters()):,}")

# =============================================================================
# Training Setup
# =============================================================================

import torch
import torch.optim as optim

# Create optimizer
optimizer = optim.Adam(fusion_system.parameters(), lr=1e-4)

# Or use different learning rates for different components
param_groups = fusion_system.get_parameter_groups()
optimizer = optim.Adam([
    {'params': param_groups[0]['params'], 'lr': 1e-4, 'name': 'swintexco'},
    {'params': param_groups[1]['params'], 'lr': 1e-4, 'name': 'fusion'}
])

print("\n✅ Optimizer created")
print("   Ready to start training!")

# =============================================================================
# Training Loop (Placeholder)
# =============================================================================

print("\n" + "="*80)
print("Training Configuration Summary")
print("="*80)
print(f"Strategy: {'Pretrained SwinTExCo' if USE_PRETRAINED_SWINTEXCO else 'Random NonLocalNet + Pretrained Swin'}")
print(f"MemFlow: Frozen (pretrained)")
print(f"SwinTExCo embed_net: Frozen (pretrained Swin backbone)")
print(f"SwinTExCo nonlocal_net: {'Fine-tuning' if USE_PRETRAINED_SWINTEXCO else 'Training from scratch'}")
print(f"SwinTExCo colornet: Frozen")
print(f"Fusion UNet: Training from scratch")
print("="*80)

# Your training loop would go here
# For example:
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         optimizer.zero_grad()
#         output = fusion_system.forward_sequence(...)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
