import torch
import torch.nn as nn
from networks.models.unetr_pp.network_architecture.tumor.unetr_pp_tumor import UNETR_PP

def test_unetr_pp():
    # Create a dummy input tensor with the shape (batch_size, channels, depth, height, width)
    input_tensor = torch.randn(1, 4, 128, 128, 128)  # Example shape for BraTS dataset

    # Initialize the UNETR_PP model
    model = UNETR_PP(
        in_channels=4,
        out_channels=3,
        feature_size=16,
        hidden_size=256,
        num_heads=4,
        pos_embed="perceptron",
        do_ds=False, 
        dims=[32, 64, 128, 256],
        depths=[2, 2, 2, 2],)


    # Forward pass through the model
    output = model(input_tensor)

    # Check the output shape
    assert output.shape == (1, 3, 128, 128, 128), "Output shape mismatch"
    print("UNETR_PP test passed successfully!")

if __name__ == "__main__":
    test_unetr_pp()
    print("All tests passed!")