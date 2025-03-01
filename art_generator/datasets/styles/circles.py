"""
Circle-based geometric pattern generator.

Creates compositions with circles, concentric forms, and circular arrangements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os
from typing import Tuple, List, Dict, Any, Optional
from art_generator.utils.color_utils import normalize_color

def generate_circles(
    seed: int, 
    size: Tuple[int, int] = (512, 512),
    min_circles: int = 10,
    max_circles: int = 30,
    circle_count: int = 0,  # For fixed count instead of min/max
    min_radius: float = 0.02,
    max_radius: float = 0.15,
    color_palette: Optional[List] = None,
    line_width: float = 0.1,
    fill_probability: float = 0.7,
    concentric_probability: float = 0.3,
    output_path: Optional[str] = None,
    symmetry: bool = False,
    ordered: bool = False,
    allow_overlap: bool = True
) -> plt.Figure:
    """
    Generate a circle-based geometric pattern.
    
    Args:
        seed: Random seed for reproducibility
        size: Size of the output image (width, height)
        min_circles: Minimum number of circles to generate
        max_circles: Maximum number of circles to generate
        circle_count: Fixed number of circles (if > 0, overrides min/max)
        min_radius: Minimum radius of circles as a fraction of image size
        max_radius: Maximum radius of circles as a fraction of image size
        color_palette: List of colors to use (if None, uses black/white)
        line_width: Width of circle lines
        fill_probability: Probability of filling a circle
        concentric_probability: Probability of generating concentric circles
        output_path: If provided, saves the image to this path
        symmetry: Creates symmetric patterns when True
        ordered: Creates more ordered, regular patterns when True
        allow_overlap: Allows overlapping elements when True
        
    Returns:
        matplotlib Figure object
    """
    np.random.seed(seed)
    
    # Default black and white palette
    if color_palette is None:
        color_palette = ["black", "white"]
    
    # Normalize color palette if needed
    normalized_palette = [normalize_color(color) for color in color_palette]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    
    # Set background color (last color in palette)
    fig.patch.set_facecolor(normalized_palette[-1])
    
    # Determine number of base circles
    num_base_circles = np.random.randint(min_circles, max_circles + 1)
    
    # Generate circle centers
    centers = np.random.uniform(0, 1, (num_base_circles, 2))
    
    # Generate circles
    for i, (x, y) in enumerate(centers):
        # Decide if we're making concentric circles
        is_concentric = np.random.random() < concentric_probability
        
        if is_concentric:
            # Generate 2-4 concentric circles
            num_concentric = np.random.randint(2, 5)
            radii = np.linspace(max_radius, min_radius, num_concentric)
            
            for radius in radii:
                # Decide whether to fill this circle
                if np.random.random() < fill_probability:
                    color_idx = np.random.randint(0, len(normalized_palette) - 1)  # Exclude background color
                    color = normalized_palette[color_idx]
                    circle = plt.Circle((x, y), radius, fill=True, 
                                      edgecolor='black', facecolor=color, 
                                      linewidth=line_width)
                else:
                    # Just add the outline
                    circle = plt.Circle((x, y), radius, fill=False, 
                                      edgecolor='black', linewidth=line_width)
                
                ax.add_patch(circle)
        else:
            # Just a single circle
            radius = np.random.uniform(min_radius, max_radius)
            
            # Decide whether to fill this circle
            if np.random.random() < fill_probability:
                color_idx = np.random.randint(0, len(normalized_palette) - 1)  # Exclude background color
                color = normalized_palette[color_idx]
                circle = plt.Circle((x, y), radius, fill=True, 
                                  edgecolor='black', facecolor=color, 
                                  linewidth=line_width)
            else:
                # Just add the outline
                circle = plt.Circle((x, y), radius, fill=False, 
                                  edgecolor='black', linewidth=line_width)
            
            ax.add_patch(circle)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    
    return fig 