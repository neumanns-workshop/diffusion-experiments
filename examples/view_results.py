"""
Utility script to view and compare training data and generated images.

This script helps visualize:
1. Original training data samples
2. Generated images from the trained model
3. Training progress samples
"""

import os
import argparse
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob

def display_images(image_paths, cols=4, figsize=(15, 10), title=None):
    """Display a grid of images from the given paths."""
    # Determine the number of rows based on the number of images and columns
    n = len(image_paths)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Load and display images
    for i, ax in enumerate(axes):
        if i < n:
            try:
                img = Image.open(image_paths[i])
                ax.imshow(img)
                ax.set_title(os.path.basename(image_paths[i]))
            except Exception as e:
                print(f"Error loading {image_paths[i]}: {e}")
                ax.text(0.5, 0.5, f"Error loading\n{os.path.basename(image_paths[i])}", 
                       ha='center', va='center')
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig

def view_results(demo_dir, style_name=None, sample_count=12):
    """
    View and compare results from a demo run.
    
    Args:
        demo_dir: Base directory of the demo output
        style_name: Name of the style to view (if None, infer from directory)
        sample_count: Number of sample images to show
    """
    # If style_name is not provided, try to infer it
    if style_name is None:
        dataset_dirs = os.listdir(os.path.join(demo_dir, "datasets"))
        if len(dataset_dirs) == 0:
            print("No datasets found in the demo directory.")
            return
        style_name = dataset_dirs[0]
        print(f"Using style: {style_name}")
    
    # Define paths
    dataset_dir = os.path.join(demo_dir, "datasets", style_name)
    model_dir = os.path.join(demo_dir, "models", style_name)
    generated_dir = os.path.join(demo_dir, "generated", style_name)
    
    # Check if directories exist
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    # View training data samples
    dataset_images = glob.glob(os.path.join(dataset_dir, "*.png"))
    if dataset_images:
        print(f"Found {len(dataset_images)} training images.")
        # Select a random subset
        if len(dataset_images) > sample_count:
            dataset_samples = random.sample(dataset_images, sample_count)
        else:
            dataset_samples = dataset_images
        
        fig1 = display_images(dataset_samples, cols=4, title="Training Data Samples")
        plt.show()
    else:
        print("No training images found.")
    
    # View generated images if available
    if os.path.exists(generated_dir):
        generated_images = glob.glob(os.path.join(generated_dir, "*.png"))
        if generated_images:
            print(f"Found {len(generated_images)} generated images.")
            # Filter out grid.png if it exists
            generated_images = [img for img in generated_images if os.path.basename(img) != "grid.png"]
            
            # Select a random subset
            if len(generated_images) > sample_count:
                generated_samples = random.sample(generated_images, sample_count)
            else:
                generated_samples = generated_images
            
            fig2 = display_images(generated_samples, cols=4, title="Generated Images")
            plt.show()
            
            # Check if grid.png exists
            grid_path = os.path.join(generated_dir, "grid.png")
            if os.path.exists(grid_path):
                plt.figure(figsize=(10, 10))
                img = plt.imread(grid_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title("Generated Images Grid")
                plt.show()
        else:
            print("No generated images found.")
    else:
        print(f"Generated directory not found: {generated_dir}")
    
    # View training progress
    samples_dir = os.path.join(model_dir, "samples")
    if os.path.exists(samples_dir):
        sample_images = sorted(glob.glob(os.path.join(samples_dir, "*.png")))
        if sample_images:
            print(f"Found {len(sample_images)} training progress samples.")
            fig3 = display_images(sample_images, cols=2, figsize=(12, len(sample_images)*3), 
                                title="Training Progress")
            plt.show()
            
            # Display losses if available
            losses_path = os.path.join(model_dir, "losses.png")
            if os.path.exists(losses_path):
                plt.figure(figsize=(10, 6))
                img = plt.imread(losses_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title("Training Losses")
                plt.show()
        else:
            print("No training progress samples found.")
    else:
        print(f"Samples directory not found: {samples_dir}")

def main():
    """Parse command line arguments and view results."""
    parser = argparse.ArgumentParser(description="View and compare demo results")
    parser.add_argument("--demo-dir", type=str, required=True,
                       help="Base directory of the demo output")
    parser.add_argument("--style", type=str, default=None,
                       help="Name of the style to view (if not provided, infer from directory)")
    parser.add_argument("--samples", type=int, default=12,
                       help="Number of sample images to show")
    
    args = parser.parse_args()
    view_results(args.demo_dir, args.style, args.samples)

if __name__ == "__main__":
    import numpy as np  # Required for array handling
    main() 