"""
Dataset generation script for geometric art.

This script generates datasets with various styles of geometric patterns.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt

from art_generator.datasets.styles import (
    generate_grid,
    generate_triangles,
    generate_circles,
    generate_brutalist
)

# Map of style names to generator functions
STYLE_GENERATORS = {
    "grid": generate_grid,
    "triangles": generate_triangles,
    "circles": generate_circles, 
    "brutalist": generate_brutalist
}

def generate_dataset(
    style: str,
    count: int,
    output_dir: str,
    seed: Optional[int] = None,
    image_size: tuple = (512, 512),
    color_palette: Optional[List[str]] = None,
    **style_params
) -> None:
    """
    Generate a dataset of geometric patterns with the specified style.
    
    Args:
        style: The style of geometric patterns to generate
        count: Number of images to generate
        output_dir: Directory to save the generated images
        seed: Random seed for reproducibility
        image_size: Size of the generated images
        color_palette: List of colors to use
        **style_params: Additional parameters specific to the style
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get the generator function for the specified style
    if style not in STYLE_GENERATORS:
        raise ValueError(f"Unknown style: {style}. Available styles: {list(STYLE_GENERATORS.keys())}")
    
    generator_fn = STYLE_GENERATORS[style]
    
    # Generate the images
    for i in tqdm(range(count), desc=f"Generating {style} dataset"):
        image_seed = seed + i if seed is not None else np.random.randint(0, 100000)
        output_path = os.path.join(output_dir, f"{style}_{i:04d}.png")
        
        # Generate the image
        fig = generator_fn(
            seed=image_seed,
            size=image_size,
            color_palette=color_palette,
            output_path=output_path,
            **style_params
        )
        
        # Close the figure to free memory
        plt.close(fig)
    
    print(f"Generated {count} {style} images in {output_dir}")

def main():
    """Main function for the dataset generation script."""
    parser = argparse.ArgumentParser(description="Generate geometric pattern datasets")
    parser.add_argument("--style", type=str, required=True, choices=STYLE_GENERATORS.keys(),
                        help="Style of geometric patterns to generate")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of images to generate")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the generated images")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of the generated images")
    parser.add_argument("--height", type=int, default=512,
                        help="Height of the generated images")
    parser.add_argument("--color-mode", type=str, default="bw", choices=["bw", "grayscale", "color"],
                        help="Color mode for the generated images")
    
    # Style-specific parameters
    parser.add_argument("--complexity", type=float, default=0.5,
                        help="Complexity of the generated patterns (0-1)")
    parser.add_argument("--fill-probability", type=float, default=0.7,
                        help="Probability of filling shapes (0-1)")
    parser.add_argument("--line-width", type=float, default=0.1,
                        help="Width of the lines in the patterns")
    
    args = parser.parse_args()
    
    # Set color palette based on color mode
    if args.color_mode == "bw":
        color_palette = ["black", "white"]
    elif args.color_mode == "grayscale":
        color_palette = ["#000000", "#1a1a1a", "#333333", "#4d4d4d", "#666666", "#808080", "#999999", "#e6e6e6", "#ffffff"]
    elif args.color_mode == "color":
        # Use a more vibrant color palette
        color_palette = ["#e63946", "#f1faee", "#a8dadc", "#457b9d", "#1d3557"]
    
    # Generate the dataset
    generate_dataset(
        style=args.style,
        count=args.count,
        output_dir=args.output,
        seed=args.seed,
        image_size=(args.width, args.height),
        color_palette=color_palette,
        complexity=args.complexity,
        fill_probability=args.fill_probability,
        line_width=args.line_width
    )

if __name__ == "__main__":
    main() 