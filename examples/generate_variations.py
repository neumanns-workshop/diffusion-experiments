"""
Generation script for creating multiple variations using different random seeds.

This script helps generate a series of variations from a trained model by:
1. Using different random seeds for the latent space
2. Creating a grid of all generated variations
3. Saving both individual images and the grid
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from art_generator.models.gan import DCGAN
from art_generator.utils.visualization import save_image_grid

def generate_variations(
    model_path: str,
    output_dir: str,
    count: int = 16,
    seed_start: int = 0,
    latent_dim: int = None,
    grid_output: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generate multiple image variations using different random seeds.
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to save the generated images
        count: Number of variations to generate
        seed_start: Starting seed value
        latent_dim: Dimension of the latent space (if None, use model's default)
        grid_output: Path to save a grid of all generated images
        device: Device to use for generation ('cuda' or 'cpu')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}")
    gan = DCGAN.load(model_path, device=device)
    
    # If latent_dim is not provided, use the model's latent_dim
    if latent_dim is None:
        latent_dim = gan.latent_dim
    
    # Generate variations with different seeds
    print(f"Generating {count} variations...")
    generated_imgs = []
    
    for i in tqdm(range(count)):
        # Set seed for reproducibility
        seed = seed_start + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random noise
        noise = torch.randn(1, latent_dim, device=device)
        
        # Generate image
        with torch.no_grad():
            img = gan.generate(1, noise)[0]
        
        # Store the image
        generated_imgs.append(img.detach().cpu())
        
        # Denormalize from [-1, 1] to [0, 1]
        img_norm = img * 0.5 + 0.5
        
        # Convert to numpy and transpose if needed
        if gan.channels == 1:
            img_np = img_norm.squeeze().cpu().numpy()
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        else:
            img_np = img_norm.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        
        # Save image with seed in filename
        output_path = os.path.join(output_dir, f"variation_seed_{seed:04d}.png")
        pil_img.save(output_path)
    
    # Save grid of all variations if requested
    if grid_output:
        print(f"Creating grid of all variations...")
        # Stack all images
        all_imgs = torch.stack(generated_imgs)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(count)))
        
        save_image_grid(
            images=all_imgs,
            output_path=grid_output,
            rows=grid_size,
            cols=grid_size,
            figsize=(15, 15),
            title=f"Variations (Seeds {seed_start} to {seed_start + count - 1})",
            cmap="gray" if gan.channels == 1 else None
        )
    
    print(f"Generated {count} variations in {output_dir}")
    if grid_output:
        print(f"Grid saved to {grid_output}")

def main():
    """Parse command line arguments and generate variations."""
    parser = argparse.ArgumentParser(description="Generate variations using different random seeds")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save the generated variations")
    parser.add_argument("--count", type=int, default=16,
                        help="Number of variations to generate")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed value")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Dimension of the latent space (if not specified, use the model's latent_dim)")
    parser.add_argument("--grid", type=str, default=None,
                        help="Path to save a grid of all variations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for generation ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Generate variations
    generate_variations(
        model_path=args.model,
        output_dir=args.output,
        count=args.count,
        seed_start=args.seed_start,
        latent_dim=args.latent_dim,
        grid_output=args.grid,
        device=args.device
    )

if __name__ == "__main__":
    main() 