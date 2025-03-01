"""
Configuration utilities for managing style settings.
"""

import os
import json
import copy
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict

@dataclass
class StyleConfig:
    """
    Configuration for a geometric style.
    """
    # Basic parameters
    style: str
    name: str  # A name for this specific style configuration
    description: str = ""
    
    # Image parameters
    image_size: int = 512
    channels: int = 1  # 1 for grayscale, 3 for RGB
    
    # Style-specific parameters
    color_palette: List[str] = field(default_factory=lambda: ["black", "white"])
    complexity: float = 0.5
    fill_probability: float = 0.7
    line_width: float = 0.1
    
    # Style-specific parameters - Grid
    grid_size: int = 5
    
    # Style-specific parameters - Triangles
    min_triangles: int = 10
    max_triangles: int = 30
    regular_triangles: float = 0.3
    triangle_count: int = 0  # For fixed count instead of min/max
    
    # Style-specific parameters - Circles
    min_circles: int = 10
    max_circles: int = 30
    min_radius: float = 0.02
    max_radius: float = 0.15
    concentric_probability: float = 0.3
    circle_count: int = 0  # For fixed count instead of min/max
    
    # Style-specific parameters - Brutalist
    min_blocks: int = 5
    max_blocks: int = 15
    block_count: int = 0  # For fixed count instead of min/max
    min_block_size: int = 30
    max_block_size: int = 50
    
    # Common structure parameters
    symmetry: bool = False
    ordered: bool = False
    allow_overlap: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "StyleConfig":
        """Create a config from a dictionary."""
        return cls(**config_dict)
    
    def get_style_params(self) -> Dict[str, Any]:
        """
        Get parameters specific to the style.
        
        Returns:
            Dictionary of style-specific parameters
        """
        # Common parameters for all styles
        common_params = {
            "complexity": self.complexity,
            "fill_probability": self.fill_probability,
            "line_width": self.line_width,
            "symmetry": self.symmetry,
            "ordered": self.ordered,
            "allow_overlap": self.allow_overlap
        }
        
        if self.style == "grid":
            return {
                **common_params,
                "grid_size": self.grid_size
            }
        elif self.style == "triangles":
            params = {
                **common_params,
                "regular_triangles": self.regular_triangles
            }
            # Use either fixed count or min/max
            if self.triangle_count > 0:
                params["triangle_count"] = self.triangle_count
            else:
                params["min_triangles"] = self.min_triangles
                params["max_triangles"] = self.max_triangles
            return params
        elif self.style == "circles":
            params = {
                **common_params,
                "concentric_probability": self.concentric_probability,
                "min_radius": self.min_radius,
                "max_radius": self.max_radius
            }
            # Use either fixed count or min/max
            if self.circle_count > 0:
                params["circle_count"] = self.circle_count
            else:
                params["min_circles"] = self.min_circles
                params["max_circles"] = self.max_circles
            return params
        elif self.style == "brutalist":
            params = {
                **common_params,
                "min_block_size": self.min_block_size,
                "max_block_size": self.max_block_size
            }
            # Use either fixed count or min/max
            if self.block_count > 0:
                params["block_count"] = self.block_count
            else:
                params["min_blocks"] = self.min_blocks
                params["max_blocks"] = self.max_blocks
            return params
        else:
            # Generic parameters for unknown styles
            return common_params

def save_config(config: StyleConfig, path: str) -> None:
    """
    Save a configuration to a JSON file.
    
    Args:
        config: The configuration to save
        path: Path to save the configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save to file
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

def load_config(path: str) -> StyleConfig:
    """
    Load a configuration from a JSON file.
    
    Args:
        path: Path to load the configuration from
        
    Returns:
        Loaded configuration
    """
    with open(path, "r") as f:
        config_dict = json.load(f)
    
    return StyleConfig.from_dict(config_dict)

# Predefined style configurations
PREDEFINED_CONFIGS = {
    "grid_mondrian": StyleConfig(
        style="grid",
        name="grid_mondrian",
        description="Mondrian-inspired grid compositions with primary colors",
        color_palette=["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"],
        grid_size=4,
        complexity=0.3,
        fill_probability=0.8
    ),
    "triangles_minimal": StyleConfig(
        style="triangles",
        name="triangles_minimal",
        description="Minimalist triangular compositions in black and white",
        color_palette=["#000000", "#FFFFFF"],
        min_triangles=5,
        max_triangles=15,
        regular_triangles=0.7,
        fill_probability=0.6
    ),
    "circles_organic": StyleConfig(
        style="circles",
        name="circles_organic",
        description="Organic circular compositions with natural colors",
        color_palette=["#2E4600", "#486B00", "#A2C523", "#7D4427", "#F8F4E3"],
        channels=3,
        min_circles=15,
        max_circles=40,
        concentric_probability=0.4,
        fill_probability=0.75
    ),
    "brutalist_concrete": StyleConfig(
        style="brutalist",
        name="brutalist_concrete",
        description="Bold, heavy brutalist compositions with concrete-like colors",
        color_palette=["#444444", "#666666", "#888888", "#aaaaaa", "#cccccc", "#eeeeee"],
        min_blocks=7,
        max_blocks=12,
        complexity=0.8,
        fill_probability=0.9
    )
} 