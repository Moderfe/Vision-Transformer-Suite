import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_data_transforms(image_size=224):
    """
    Defines standard image transformations for training and validation.

    Args:
        image_size (int): The target size for resizing and cropping images.

    Returns:
        dict: A dictionary containing 'train' and 'val' transformations.
    """
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    logging.info(f"Data transformations defined for image size: {image_size}")
    return data_transforms

def get_data_loaders(data_dir, image_size=224, batch_size=32, num_workers=4):
    """
    Creates data loaders for training and validation datasets.

    Args:
        data_dir (str): Path to the root directory containing 'train' and 'val' subdirectories.
        image_size (int): Target image size.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        dict: A dictionary containing 'train' and 'val' data loaders.
    """
    data_transforms = get_data_transforms(image_size)
    image_datasets = {
        x: ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    data_loaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    logging.info(f"Data loaders created for {data_dir}. Train size: {dataset_sizes["train"]}, Val size: {dataset_sizes["val"]}")
    return data_loaders, dataset_sizes, class_names

if __name__ == "__main__":
    # This part requires dummy data directories to run without error
    # For demonstration, we will just log the creation of transforms.
    print("\n--- Vision Transformer Suite Utils Example ---")
    
    # Create dummy directories for demonstration
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

    data_loaders, dataset_sizes, class_names = get_data_loaders(dummy_data_dir, batch_size=4)

    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Class names: {class_names}")

    # Get a batch of training data
    inputs, labels = next(iter(data_loaders["train"]))
    print(f"Batch input shape: {inputs.shape}")
    print(f"Batch labels: {labels}")

    # Clean up dummy directories
    import shutil
    shutil.rmtree(dummy_data_dir)
    logging.info(f"Cleaned up dummy data directory: {dummy_data_dir}")

    print("\n--- Vision Transformer Suite Utils example finished ---")
