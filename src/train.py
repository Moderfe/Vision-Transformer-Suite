import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from src.utils import get_data_loaders
import time
import copy
import logging
import os

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

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25, device="cpu"):
    """
    Trains the given model using the provided data loaders.

    Args:
        model (nn.Module): The model to be trained.
        dataloaders (dict): Dictionary of data loaders for training and validation.
        dataset_sizes (dict): Dictionary of dataset sizes for training and validation.
        criterion (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimizer.
        num_epochs (int): Number of epochs to train for.
        device (str): Device to run training on (e.g., "cuda" or "cpu").

    Returns:
        nn.Module: The trained model.
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.to(device)

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch}/{num_epochs - 1}")
        logging.info("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # deep copy the model if it's the best accuracy
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logging.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    logging.info(f"Best val Acc: {best_acc:.4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    print("\n--- Vision Transformer Suite Training Example ---")

    # Create dummy data directories for demonstration
    dummy_data_dir = "./dummy_data"
    os.makedirs(os.path.join(dummy_data_dir, "train", "class_a"), exist_ok=True)
    os.makedirs(os.path.join(dummy_data_dir, "train", "class_b"), exist_ok=True)
    os.makedirs(os.path.join(dummy_data_dir, "val", "class_a"), exist_ok=True)
    os.makedirs(os.path.join(dummy_data_dir, "val", "class_b"), exist_ok=True)

    # Create dummy image files
    from PIL import Image
    for i in range(5):
        Image.new("RGB", (100, 100), color = (i*50, 0, 0)).save(os.path.join(dummy_data_dir, "train", "class_a", f"img_{i}.png"))
        Image.new("RGB", (100, 100), color = (0, i*50, 0)).save(os.path.join(dummy_data_dir, "train", "class_b", f"img_{i}.png"))
        Image.new("RGB", (100, 100), color = (i*50, i*50, 0)).save(os.path.join(dummy_data_dir, "val", "class_a", f"val_img_{i}.png"))
        Image.new("RGB", (100, 100), color = (0, i*50, i*50)).save(os.path.join(dummy_data_dir, "val", "class_b", f"val_img_{i}.png"))

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    dataloaders, dataset_sizes, class_names = get_data_loaders(dummy_data_dir, batch_size=2, num_workers=0) # num_workers=0 for Windows compatibility
    num_classes = len(class_names)

    # Initialize model
    model = VisionTransformerWrapper(model_name="resnet50", pretrained=True, num_classes=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=2, device=device)

    # Save the trained model
    torch.save(model.state_dict(), "best_model.pth")
    logging.info("Trained model saved to best_model.pth")

    # Clean up dummy directories
    import shutil
    shutil.rmtree(dummy_data_dir)
    logging.info(f"Cleaned up dummy data directory: {dummy_data_dir}")

    print("\n--- Vision Transformer Suite Training example finished ---")
