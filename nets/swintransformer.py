import torch
import timm
import torch.nn as nn


class SwinTransformerV2(nn.Module):
    def __init__(self, model_name='swinv2_cr_tiny_ns_224', pretrained=True):
        super(SwinTransformerV2, self).__init__()
        # Load the Swin Transformer model from timm
        self.features = timm.create_model(model_name, pretrained=pretrained)
        self.features_only = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        # Select specific stages (blocks) for low-level and high-level features
        self.low_level_idx = 0  # Index for low-level features (early layers)
        self.high_level_idx = -1  # Index for high-level features (last layer)

    def forward(self, x):
        # Forward pass through the Swin Transformer backbone
        extract_features = self.features_only(x)

        # Extract low-level and high-level features
        low_level_features = extract_features[self.low_level_idx]  # Early features
        high_level_features = extract_features[self.high_level_idx]  # Final features

        return low_level_features, high_level_features



def swintransformerv2(pretrained=True, downsample_factor=16):
    model_name = 'swinv2_cr_tiny_ns_224'
    if pretrained:
        model = SwinTransformerV2(model_name, pretrained=True)
    return model




class SwinTransformer_base(nn.Module):
    def __init__(self, model_name='swin_s3_base_224', pretrained=True):
        super(SwinTransformer_base, self).__init__()
        # Load the Swin Transformer model from timm
        self.features = timm.create_model(model_name, pretrained=pretrained)
        self.features_only = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        # Select specific stages (blocks) for low-level and high-level features
        self.low_level_idx = 0  # Index for low-level features (early layers)
        self.high_level_idx = -1  # Index for high-level features (last layer)

    def forward(self, x):
        # Forward pass through the Swin Transformer backbone
        extract_features = self.features_only(x)

        # Extract low-level and high-level features
        low_level_features = extract_features[self.low_level_idx]  # Early features
        high_level_features = extract_features[self.high_level_idx]  # Final features

        # Reshape the tensors to match the same format
        # Change (Batch, Height, Width, Channels) -> (Batch, Channels, Height, Width)
        if low_level_features.dim() == 4 and low_level_features.shape[-1] != low_level_features.shape[1]:
            low_level_features = low_level_features.permute(0, 3, 1, 2)

        if high_level_features.dim() == 4 and high_level_features.shape[-1] != high_level_features.shape[1]:
            high_level_features = high_level_features.permute(0, 3, 1, 2)

        return low_level_features, high_level_features



def swintransformerv2(pretrained=True, downsample_factor=16):
    model_name = 'swinv2_cr_tiny_ns_224'
    if pretrained:
        model = SwinTransformerV2(model_name, pretrained=True)
    return model


def swintransformer_base(pretrained=True, downsample_factor=16):
    model_name = 'swin_s3_base_224'
    if pretrained:
        model = SwinTransformer_base(model_name, pretrained=True)
    return model


# # Example usage
# model = SwinTransformerV2(pretrained=True)
# input_tensor = torch.randn(1, 3, 224, 224)  # Batch of 1 image (3 channels, 224x224 resolution)
# low_features, high_features = model(input_tensor)
#
# print(f"Low-level feature shape: {low_features.shape}")
# print(f"High-level feature shape: {high_features.shape}")
#
#
# model2 = SwinTransformer_base(pretrained=True)
# input_tensor2 = torch.randn(1, 3, 224, 224)  # Batch of 1 image (3 channels, 224x224 resolution)
# low_features2, high_features2 = model(input_tensor2)
#
# print(f"Low-level feature shape: {low_features2.shape}")
# print(f"High-level feature shape: {high_features2.shape}")