"""
Grid-based geometric pattern generator.

Creates Mondrian-like compositions with grid-based rectangles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os
from typing import Tuple, List, Dict, Any, Optional, Union
from art_generator.utils.color_utils import normalize_color

def generate_grid(
    seed: int, 
    size: Tuple[int, int] = (512, 512),
    grid_size: int = 5,
    color_palette: Optional[List] = None,
    line_width: float = 0.1,
    fill_probability: float = 0.8,
    complexity: float = 0.5,
    output_path: Optional[str] = None,
    symmetry: bool = False,
    ordered: bool = False,
    allow_overlap: bool = True
) -> plt.Figure:
    """
    Generate a grid-based geometric pattern.
    
    Args:
        seed: Random seed for reproducibility
        size: Size of the output image (width, height)
        grid_size: Base size of the grid units
        color_palette: List of colors to use (if None, uses black/white)
        line_width: Width of grid lines
        fill_probability: Probability of filling a cell
        complexity: Controls the complexity of the composition (0-1)
        output_path: If provided, saves the image to this path
        symmetry: Creates symmetric patterns when True
        ordered: Creates more ordered, regular grid patterns when True
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
    
    # Determine grid divisions based on complexity
    base_divisions = max(2, int(grid_size * (1 + complexity)))
    
    # Create grid - more regular when ordered is True
    if ordered:
        # Create a perfectly regular grid
        x_divisions = np.linspace(0, 1, base_divisions)
        y_divisions = np.linspace(0, 1, base_divisions)
    else:
        # Create grid with some randomness
        x_divisions = np.linspace(0, 1, base_divisions)
        y_divisions = np.linspace(0, 1, base_divisions)
        
        # Add some randomness to grid lines based on complexity
        if complexity > 0.3:
            # Add some random divisions
            num_extra_x = int(base_divisions * complexity * 0.5)
            num_extra_y = int(base_divisions * complexity * 0.5)
            
            extra_x = np.random.uniform(0, 1, num_extra_x)
            extra_y = np.random.uniform(0, 1, num_extra_y)
            
            x_divisions = np.sort(np.concatenate([x_divisions, extra_x]))
            y_divisions = np.sort(np.concatenate([y_divisions, extra_y]))
    
    # For symmetric patterns, ensure we have a center point
    if symmetry:
        if 0.5 not in x_divisions:
            x_divisions = np.sort(np.append(x_divisions, 0.5))
        if 0.5 not in y_divisions:
            y_divisions = np.sort(np.append(y_divisions, 0.5))
    
    # Track filled areas if we don't allow overlap
    filled_areas = []
    
    # Handle symmetry
    if symmetry:
        # Only process first quadrant or half for symmetry
        mid_x_idx = np.where(x_divisions >= 0.5)[0][0]
        mid_y_idx = np.where(y_divisions >= 0.5)[0][0]
        
        # Generate patterns for first quadrant
        for i in range(mid_x_idx):
            for j in range(mid_y_idx):
                x = x_divisions[i]
                y = y_divisions[j]
                width = x_divisions[i+1] - x
                height = y_divisions[j+1] - y
                
                # Decide whether to fill this rectangle
                if np.random.random() < fill_probability:
                    color_idx = np.random.randint(0, len(normalized_palette) - 1)  # Exclude background color
                    color = normalized_palette[color_idx]
                    
                    # Create symmetric rectangles in all four quadrants
                    # First quadrant (original)
                    rect1 = plt.Rectangle((x, y), width, height, fill=True,
                                        edgecolor='black', facecolor=color,
                                        linewidth=line_width)
                    ax.add_patch(rect1)
                    
                    # Second quadrant (reflected across x-axis)
                    x2 = 1 - x - width
                    rect2 = plt.Rectangle((x2, y), width, height, fill=True,
                                        edgecolor='black', facecolor=color,
                                        linewidth=line_width)
                    ax.add_patch(rect2)
                    
                    # Third quadrant (reflected across y-axis)
                    y2 = 1 - y - height
                    rect3 = plt.Rectangle((x, y2), width, height, fill=True,
                                        edgecolor='black', facecolor=color,
                                        linewidth=line_width)
                    ax.add_patch(rect3)
                    
                    # Fourth quadrant (reflected across both axes)
                    rect4 = plt.Rectangle((x2, y2), width, height, fill=True,
                                        edgecolor='black', facecolor=color,
                                        linewidth=line_width)
                    ax.add_patch(rect4)
                else:
                    # Just add the outlines with symmetry
                    rect1 = plt.Rectangle((x, y), width, height, fill=False,
                                        edgecolor='black', linewidth=line_width)
                    ax.add_patch(rect1)
                    
                    x2 = 1 - x - width
                    rect2 = plt.Rectangle((x2, y), width, height, fill=False,
                                        edgecolor='black', linewidth=line_width)
                    ax.add_patch(rect2)
                    
                    y2 = 1 - y - height
                    rect3 = plt.Rectangle((x, y2), width, height, fill=False,
                                        edgecolor='black', linewidth=line_width)
                    ax.add_patch(rect3)
                    
                    rect4 = plt.Rectangle((x2, y2), width, height, fill=False,
                                        edgecolor='black', linewidth=line_width)
                    ax.add_patch(rect4)
    else:
        # Original asymmetric logic
        for i in range(len(x_divisions) - 1):
            for j in range(len(y_divisions) - 1):
                x = x_divisions[i]
                y = y_divisions[j]
                width = x_divisions[i+1] - x
                height = y_divisions[j+1] - y
                
                # Check for overlap if not allowed
                if not allow_overlap:
                    # Define current rectangle
                    current_rect = (x, y, x + width, y + height)
                    
                    # Check if it overlaps with any filled areas
                    overlaps = False
                    for area in filled_areas:
                        # Simple overlap check
                        if not (current_rect[2] <= area[0] or current_rect[0] >= area[2] or
                                current_rect[3] <= area[1] or current_rect[1] >= area[3]):
                            overlaps = True
                            break
                    
                    if overlaps:
                        continue
                
                # Decide whether to fill this rectangle
                if np.random.random() < fill_probability:
                    color_idx = np.random.randint(0, len(normalized_palette) - 1)  # Exclude background color
                    color = normalized_palette[color_idx]
                    rect = plt.Rectangle((x, y), width, height, fill=True, 
                                        edgecolor='black', facecolor=color, 
                                        linewidth=line_width)
                    ax.add_patch(rect)
                    
                    # Track this filled area if overlap is not allowed
                    if not allow_overlap:
                        filled_areas.append(current_rect)
                else:
                    # Just add the outline
                    rect = plt.Rectangle((x, y), width, height, fill=False, 
                                        edgecolor='black', linewidth=line_width)
                    ax.add_patch(rect)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    
    return fig 