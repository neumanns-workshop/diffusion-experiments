"""
Brutalist geometric pattern generator.

Creates bold, chunky architectural forms inspired by brutalist design.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import os
from typing import Tuple, List, Dict, Any, Optional
from art_generator.utils.color_utils import normalize_color

def generate_brutalist(
    seed: int, 
    size: Tuple[int, int] = (512, 512),
    min_blocks: int = 5,
    max_blocks: int = 15,
    block_count: int = 0,  # For fixed count instead of min/max
    min_block_size: int = 30,
    max_block_size: int = 50,
    color_palette: Optional[List] = None,
    line_width: float = 0.1,
    complexity: float = 0.5,
    fill_probability: float = 0.9,
    output_path: Optional[str] = None,
    symmetry: bool = False,
    ordered: bool = False,
    allow_overlap: bool = True
) -> plt.Figure:
    """
    Generate a brutalist geometric pattern.
    
    Args:
        seed: Random seed for reproducibility
        size: Size of the output image (width, height)
        min_blocks: Minimum number of blocks to generate
        max_blocks: Maximum number of blocks to generate
        block_count: Fixed number of blocks (if > 0, overrides min/max)
        min_block_size: Minimum size of blocks
        max_block_size: Maximum size of blocks
        color_palette: List of colors to use (if None, uses grayscale)
        line_width: Width of block borders
        complexity: Controls the complexity of the composition (0-1)
        fill_probability: Probability of filling a block
        output_path: If provided, saves the image to this path
        symmetry: Creates symmetric patterns when True
        ordered: Creates more ordered, regular patterns when True
        allow_overlap: Allows overlapping elements when True
        
    Returns:
        matplotlib Figure object
    """
    np.random.seed(seed)
    
    # Default grayscale palette
    if color_palette is None:
        color_palette = ["#333333", "#555555", "#777777", "#999999", "#bbbbbb", "#ffffff"]
    
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
    
    # Determine number of blocks
    if block_count > 0:
        num_blocks = block_count
    else:
        num_blocks = np.random.randint(min_blocks, max_blocks + 1)
    
    # Generate base blocks (larger rectangles)
    for _ in range(num_blocks):
        # Create a block with random position and size
        x = np.random.uniform(0, 0.8)
        y = np.random.uniform(0, 0.8)
        
        # Blocks are generally larger in brutalist style
        width = np.random.uniform(0.1, 0.4)
        height = np.random.uniform(0.1, 0.4)
        
        # Ensure blocks stay within canvas
        if x + width > 1:
            width = 1 - x
        if y + height > 1:
            height = 1 - y
        
        # More likely to fill blocks in brutalist style
        if np.random.random() < fill_probability:
            color_idx = np.random.randint(0, len(normalized_palette) - 1)  # Exclude background color
            color = normalized_palette[color_idx]
            rect = plt.Rectangle((x, y), width, height, fill=True, 
                               edgecolor='black', facecolor=color, 
                               linewidth=line_width)
        else:
            # Just add the outline
            rect = plt.Rectangle((x, y), width, height, fill=False, 
                               edgecolor='black', linewidth=line_width)
        
        ax.add_patch(rect)
    
    # Add details (smaller blocks, lines) based on complexity
    if complexity > 0.3:
        num_details = int(num_blocks * complexity * 2)
        
        for _ in range(num_details):
            detail_type = np.random.choice(["small_block", "line", "window_grid"])
            
            if detail_type == "small_block":
                # Add a smaller block
                x = np.random.uniform(0, 0.9)
                y = np.random.uniform(0, 0.9)
                width = np.random.uniform(0.05, 0.15)
                height = np.random.uniform(0.05, 0.15)
                
                # Ensure blocks stay within canvas
                if x + width > 1:
                    width = 1 - x
                if y + height > 1:
                    height = 1 - y
                
                color_idx = np.random.randint(0, len(normalized_palette) - 1)  # Exclude background color
                color = normalized_palette[color_idx]
                rect = plt.Rectangle((x, y), width, height, fill=True, 
                                   edgecolor='black', facecolor=color, 
                                   linewidth=line_width)
                ax.add_patch(rect)
                
            elif detail_type == "line":
                # Add a line (concrete beam/support)
                x1 = np.random.uniform(0, 1)
                y1 = np.random.uniform(0, 1)
                x2 = np.random.uniform(0, 1)
                y2 = np.random.uniform(0, 1)
                
                line = plt.Line2D([x1, x2], [y1, y2], linewidth=line_width*3, color="black")
                ax.add_line(line)
                
            elif detail_type == "window_grid":
                # Add a grid of windows (common in brutalist buildings)
                x = np.random.uniform(0, 0.7)
                y = np.random.uniform(0, 0.7)
                width = np.random.uniform(0.2, 0.3)
                height = np.random.uniform(0.2, 0.3)
                
                # Ensure grid stays within canvas
                if x + width > 1:
                    width = 1 - x
                if y + height > 1:
                    height = 1 - y
                
                rows = np.random.randint(2, 5)
                cols = np.random.randint(2, 5)
                
                # Create window grid
                for i in range(rows):
                    for j in range(cols):
                        window_x = x + (j * width / cols)
                        window_y = y + (i * height / rows)
                        window_width = width / cols
                        window_height = height / rows
                        
                        window = plt.Rectangle((window_x, window_y), 
                                             window_width * 0.8, window_height * 0.8, 
                                             fill=True, 
                                             edgecolor='black', 
                                             facecolor=normalized_palette[0],  # Darkest color for windows
                                             linewidth=line_width)
                        ax.add_patch(window)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    
    return fig 