"""
Generate images using trained models.
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from typing import Optional, List

from art_generator.models.gan import DCGAN
from art_generator.utils.visualization import save_image_grid

def generate_images(
    model_path: str,
    output_dir: str,
    count: int = 10,
    latent_dim: Optional[int] = None,
    noise_seed: Optional[int] = None,
    grid_output: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[torch.Tensor]:
    """
    Generate images using a trained model.
    
    Args:
        model_path: Path to the trained model
        output_dir: Directory to save the generated images
        count: Number of images to generate
        latent_dim: Dimension of the latent space (if None, use the model's latent_dim)
        noise_seed: Random seed for generating noise
        grid_output: Path to save a grid of all generated images
        device: Device to use for generation ('cuda' or 'cpu')
        
    Returns:
        List of generated images as tensors
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    gan = DCGAN.load(model_path, device=device)
    
    # If latent_dim is not provided, use the model's latent_dim
    if latent_dim is None:
        latent_dim = gan.latent_dim
    
    # Set random seed if provided
    if noise_seed is not None:
        torch.manual_seed(noise_seed)
        np.random.seed(noise_seed)
    
    # Generate random noise
    noise = torch.randn(count, latent_dim, device=device)
    
    # Generate images
    with torch.no_grad():
        generated_imgs = gan.generate(count, noise)
    
    # Convert to numpy and save
    generated_imgs_cpu = generated_imgs.detach().cpu()
    
    # Save individual images
    for i, img in enumerate(generated_imgs_cpu):
        # Denormalize from [-1, 1] to [0, 1]
        img = img * 0.5 + 0.5
        
        # Convert to numpy and transpose if needed
        if gan.channels == 1:
            img_np = img.squeeze().numpy()
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        else:
            img_np = img.numpy().transpose(1, 2, 0)  # CHW -> HWC
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        
        # Save image
        pil_img.save(os.path.join(output_dir, f"generated_{i:04d}.png"))
    
    # Save grid if requested
    if grid_output:
        save_image_grid(
            images=generated_imgs_cpu,
            output_path=grid_output,
            title="Generated Images",
            cmap="gray" if gan.channels == 1 else None
        )
    
    return generated_imgs

def main():
    """Main function for generating images."""
    parser = argparse.ArgumentParser(description="Generate images using a trained model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save the generated images")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of images to generate")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Dimension of the latent space (if not specified, use the model's latent_dim)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for generating noise")
    parser.add_argument("--grid", type=str, default=None,
                        help="Path to save a grid of all generated images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for generation ('cuda' or 'cpu')")
    
    args = parser.parse_args()
    
    # Generate images
    generate_images(
        model_path=args.model,
        output_dir=args.output,
        count=args.count,
        latent_dim=args.latent_dim,
        noise_seed=args.seed,
        grid_output=args.grid,
        device=args.device
    )
    
    print(f"Generated {args.count} images in {args.output}")

if __name__ == "__main__":
    main() 