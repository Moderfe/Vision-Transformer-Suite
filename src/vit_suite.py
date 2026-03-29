import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import logging

# Configure logging for better insights into model operations
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VisionTransformerWrapper(nn.Module):
    """
    A simplified wrapper for various Vision Transformer (ViT) models.
    For demonstration, this uses a pre-trained ResNet as a stand-in for a complex ViT.
    In a real scenario, this would integrate with actual ViT implementations (e.g., from `timm` library).
    """
    def __init__(self, model_name="resnet50", pretrained=True, num_classes=1000):
        super().__init__()
        self.model_name = model_name
        logging.info(f"Initializing VisionTransformerWrapper with model: {model_name}, pretrained: {pretrained}")

        if model_name == "resnet50":
            if pretrained:
                self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.model = resnet50()
            # Modify the final layer if num_classes is different from default ImageNet (1000)
            if num_classes != 1000:
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            # Placeholder for actual ViT integration
            logging.warning(f"Model {model_name} is not directly supported in this simplified wrapper. Using ResNet50 instead.")
            if pretrained:
                self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.model = resnet50()
            if num_classes != 1000:
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        logging.info(f"Model {model_name} loaded successfully.")

    def forward(self, x):
        return self.model(x)

def create_model(model_name, pretrained=True, num_classes=1000):
    """
    Factory function to create a Vision Transformer model.

    Args:
        model_name (str): Name of the model to create (e.g., "swin_base_patch4_window7_224").
                          Currently, only "resnet50" is directly supported as a placeholder.
        pretrained (bool): Whether to load pre-trained weights.
        num_classes (int): Number of output classes for the model.

    Returns:
        VisionTransformerWrapper: An instance of the wrapped model.
    """
    logging.info(f"Creating model: {model_name} with pretrained={pretrained}, num_classes={num_classes}")
    # In a real application, this would map model_name to actual ViT architectures
    # For now, we'll use ResNet50 as a representative model.
    return VisionTransformerWrapper(model_name="resnet50", pretrained=pretrained, num_classes=num_classes)


if __name__ == "__main__":
    # Example Usage
    print("\n--- Vision Transformer Suite Example ---")

    # 1. Create a pre-trained model
    model = create_model("swin_base_patch4_window7_224", pretrained=True)
    model.eval() # Set to evaluation mode
    logging.info("Model created and set to evaluation mode.")

    # 2. Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    logging.info("Image transformations defined.")

    # 3. Create a dummy image (e.g., a red square)
    dummy_image = Image.new("RGB", (256, 256), color = "red")
    img_tensor = transform(dummy_image).unsqueeze(0) # Add batch dimension
    logging.info(f"Dummy image created with shape: {img_tensor.shape}")

    # 4. Perform inference
    with torch.no_grad():
        output = model(img_tensor)
    logging.info(f"Inference performed. Output shape: {output.shape}")

    # 5. Print a sample of the output (e.g., top 5 class probabilities)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    top5_prob, top5_idx = torch.topk(probabilities, 5)
    logging.info("Top 5 predicted probabilities and indices:")
    for i in range(top5_prob.size(0)):
        logging.info(f"  Class {top5_idx[i].item()}: {top5_prob[i].item():.4f}")

    print("\n--- Vision Transformer Suite example finished ---")
