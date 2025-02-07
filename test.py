from datasets import load_dataset
import os
from PIL import Image


def save_dataset(dataset, root_dir):
    """
    Save images from the dataset into class-based directories.
    :param dataset: Hugging Face dataset with 'image' and 'label'
    :param root_dir: Target root directory
    """
    os.makedirs(root_dir, exist_ok=True)

    for idx, sample in enumerate(dataset):
        image = sample['image']  # Assuming dataset contains images
        label = sample['label']  # Assuming dataset contains labels

        # Create class directory
        class_dir = os.path.join(root_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        # Save image
        image_path = os.path.join(class_dir, f"{idx}.JPEG")
        image.save(image_path)


def example_usage():
    dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    val_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')

    save_dataset(dataset, 'C:/Personal/UC/Research/codes/train')  # Adjust path as needed
    save_dataset(val_dataset, 'C:/Personal/UC/Research/codes/val')

if __name__ == '__main__':
    example_usage()

