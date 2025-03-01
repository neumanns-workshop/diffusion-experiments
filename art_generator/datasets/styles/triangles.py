"""
Triangle-based geometric pattern generator.

Creates tessellations and compositions using triangular forms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os
from typing import Tuple, List, Dict, Any, Optional
from art_generator.utils.color_utils import normalize_color

def generate_triangles(
    seed: int, 
    size: Tuple[int, int] = (512, 512),
    min_triangles: int = 10,
    max_triangles: int = 30,
    triangle_count: int = 0,  # For fixed count instead of min/max
    color_palette: Optional[List] = None,
    line_width: float = 0.1,
    fill_probability: float = 0.7,
    regular_triangles: float = 0.3,  # Probability of generating regular (equilateral) triangles
    output_path: Optional[str] = None,
    symmetry: bool = False,
    ordered: bool = False,
    allow_overlap: bool = True
) -> plt.Figure:
    """
    Generate a triangle-based geometric pattern.
    
    Args:
        seed: Random seed for reproducibility
        size: Size of the output image (width, height)
        min_triangles: Minimum number of triangles to generate
        max_triangles: Maximum number of triangles to generate
        triangle_count: Fixed number of triangles (if > 0, overrides min/max)
        color_palette: List of colors to use (if None, uses black/white)
        line_width: Width of triangle lines
        fill_probability: Probability of filling a triangle
        regular_triangles: Probability of generating regular triangles
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
    
    # Determine number of triangles
    if triangle_count > 0:
        num_triangles = triangle_count
    else:
        num_triangles = np.random.randint(min_triangles, max_triangles + 1)
    
    # Generate triangles
    for _ in range(num_triangles):
        # Decide if we're making a regular triangle
        is_regular = np.random.random() < regular_triangles
        
        if is_regular:
            # Create equilateral triangle
            center_x = np.random.uniform(0.2, 0.8)
            center_y = np.random.uniform(0.2, 0.8)
            size = np.random.uniform(0.05, 0.2)
            rotation = np.random.uniform(0, 2*np.pi)
            
            # Create triangle points (equilateral)
            angles = np.linspace(0, 2*np.pi, 4)[:-1] + rotation  # 3 points + close
            points = np.array([
                [center_x + size * np.cos(angle), center_y + size * np.sin(angle)]
                for angle in angles
            ])
        else:
            # Create random triangle
            points = np.random.uniform(0, 1, (3, 2))
        
        # Decide whether to fill this triangle
        if np.random.random() < fill_probability:
            color_idx = np.random.randint(0, len(normalized_palette) - 1)  # Exclude background color
            color = normalized_palette[color_idx]
            triangle = plt.Polygon(points, fill=True, 
                                 edgecolor='black', facecolor=color, 
                                 linewidth=line_width)
        else:
            # Just add the outline
            triangle = plt.Polygon(points, fill=False, 
                                 edgecolor='black', linewidth=line_width)
        
        ax.add_patch(triangle)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    
    return fig 