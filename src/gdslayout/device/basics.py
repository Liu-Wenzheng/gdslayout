import numpy as np
from typing import Tuple

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, Any


@gf.cell_with_module_name
def anchor_arc(
    radius: float = 2.0,
    width: float = 0.5,
    layer: Tuple[int, int] = (1, 0),
    resolution: float = 0.02,
) -> gf.Component:
    """
    Creates a custom arc component, now registered with gdsfactory.
    The function name 'anchor_arc' is the name used for registration.
    """
    # Calculate number of points for a smooth curve
    npoints = int(radius * np.pi / 2 / resolution)
    if npoints < 2:
        npoints = 2
        
    t = np.linspace(0, -np.pi / 2, npoints)
    y_coords = radius * np.sin(t) + radius
    x1_shifted = (radius * np.cos(t) - radius) - width / 2
    x2_shifted = -(radius * np.cos(t) - radius) + width / 2

    x_path = np.concatenate([x1_shifted[::-1], x2_shifted])
    y_path = np.concatenate([y_coords[::-1], y_coords])
    polygon_points = np.stack([x_path, y_path], axis=1)

    # Create the component and add the polygon
    c = gf.Component()
    c.add_polygon(polygon_points, layer=layer)

    # Add ports as defined in your code
    c.add_port(name="o1", center=(0, radius), width=width, orientation=90, layer=layer)
    c.add_port(name="o2", center=(0, 0), width=2*radius + width, orientation=270, layer=layer)

    return c


@gf.cell_with_module_name
def straight(
    length: float = 10.0,
    npoints: int = 2,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
    layer: Any = (1, 0),
) -> gf.Component:
    """Returns a Straight waveguide.

    Args:
        length: straight length (um).
        npoints: number of points.
        cross_section: specification (CrossSection, string or dict).
        width: width of the waveguide. If None, it will use the width of the cross_section.

    .. code::

        o1  ──────────────── o2
                length
    """
    if width is not None:
        x = gf.get_cross_section(cross_section, width=width, layer=layer)
    else:
        x = gf.get_cross_section(cross_section, layer=layer)
    p = gf.path.straight(length=length, npoints=npoints)
    c = p.extrude(x)
    x.add_bbox(c)

    c.info["length"] = length
    c.info["width"] = x.width if len(x.sections) == 0 else x.sections[0].width
    c.add_route_info(cross_section=x, length=length)
    return c