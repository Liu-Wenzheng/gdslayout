import numpy as np

import gdsfactory as gf
from gdsfactory.typings import Any

import matplotlib.pyplot as plt

def microtoroid(
    length: float = 1.0, 
    height: float = 1.0,
    radius: float = 50.0, 
    resolution: float = 1.0, 
    layer: Any = (1, 0)
):
    """
    Returns a microtoroid component.
    """
    if height > 2*radius:
        raise ValueError("Height must be less than or equal to 2*radius.")

    n = int(2*np.pi*radius/resolution)
    radian_resolution = 2*np.pi / n
    
    D = gf.Component()
    half_vertical_side = np.arange(0, height/2, resolution)
    top_side = np.arange(0, length, resolution)
    θ = np.arcsin(height/2/radius)
    n = int(np.round(θ / radian_resolution))
    half_circular_arc_y = radius * np.sin(np.linspace(0, θ, n))
    half_circular_arc_x = radius * np.cos(np.linspace(0, θ, n)) - radius*np.cos(θ)
    half_points_xpts = np.concatenate([
        half_vertical_side * 0 - length,
        top_side - length,
        half_circular_arc_x[::-1]
    ])
    half_points_ypts = np.concatenate([
        half_vertical_side,
        np.ones_like(top_side) *  height/2,
        half_circular_arc_y[::-1]
    ])
    xpts = np.concatenate([half_points_xpts, half_points_xpts[::-1]])
    ypts = np.concatenate([half_points_ypts, -half_points_ypts[::-1]])
    D.add_polygon(points=list(zip(xpts - radius*(1-np.cos(θ)), ypts)), layer=layer)
    P = gf.Path(list(zip(xpts, ypts))).rotate(90)
    D.add_port(name="coupler", center=(0, 0), width=1, orientation=0, layer=layer)
    return D, P
