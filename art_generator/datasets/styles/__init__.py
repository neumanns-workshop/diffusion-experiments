"""
Various geometric style generators.
"""

from art_generator.datasets.styles.grid import generate_grid
from art_generator.datasets.styles.triangles import generate_triangles
from art_generator.datasets.styles.circles import generate_circles
from art_generator.datasets.styles.brutalist import generate_brutalist

grid = generate_grid
triangles = generate_triangles
circles = generate_circles
brutalist = generate_brutalist

__all__ = ["generate_grid", "generate_triangles", "generate_circles", "generate_brutalist"] 