import numpy as np
from typing import Tuple

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, Any
from gdsfactory.components.containers.component_sequence import component_sequence
from . import register_device


class OptoMechanicalMass:
    """
    A class to generate a standard or inverted optomechanical mass with release holes and ports.
    """

    def __init__(
        self,
        mass_width: float = 122,
        mass_height: float = 151.5,
        outcrop_width: float = 16.32,
        outcrop_height: float = 3,
        release_hole_width: float = 2,
        release_hole_spacing_x: float = 2,
        release_hole_height: float = 6,
        release_hole_spacing_y: float = 2.75,
        spring_height: float = 10,
        port_width: float = 1,
        num_beams: int = 2,
        mass_padding: float = 2,
        layer: Any = (1,0),
    ):

        self.mass_width = mass_width
        self.mass_height = mass_height
        self.outcrop_width = outcrop_width
        self.outcrop_height = outcrop_height
        self.release_hole_width = release_hole_width
        self.release_hole_spacing_x = release_hole_spacing_x
        self.release_hole_height = release_hole_height
        self.release_hole_spacing_y = release_hole_spacing_y
        self.spring_height = spring_height
        self.port_width = port_width
        self.num_beams = max(2, num_beams)  # Ensure at least 2 beams
        self.mass_padding = mass_padding
        self.layer = layer

    def _create_base_mass(self) -> gf.Component:
        """Creates the solid rectangular mass with its top outcrop."""
        c = gf.Component()
        c.add_ref(
            gf.components.rectangle(
                size=(self.mass_width, self.mass_height),
                layer=self.layer,
                centered=True,
            )
        )
        if self.outcrop_width > 0 and self.outcrop_height > 0:
            outcrop = c.add_ref(
                gf.components.rectangle(
                    size=(self.outcrop_width, self.outcrop_height),
                    layer=self.layer,
                    centered=True,
                )
            )
            outcrop.dmovey(self.mass_height / 2 + self.outcrop_height / 2)
        return c

    def _create_release_holes(self) -> gf.Component:
        """
        Creates an array of rectangular release holes.
        This method is rewritten to avoid the gf.components.array TypeError.
        """
        c = gf.Component()
        hole = gf.components.rectangle(
            size=(self.release_hole_width, self.release_hole_height),
            layer=self.layer,
            centered=True,
        )

        # Calculate pitch (center-to-center distance)
        pitch_x = self.release_hole_width + self.release_hole_spacing_x
        pitch_y = self.release_hole_height + self.release_hole_spacing_y

        # Calculate number of holes in each direction
        # We leave at least half a spacing on each side
        nx = int((self.mass_width - self.release_hole_spacing_x * 2) / pitch_x) + 1
        ny = int((self.mass_height - self.release_hole_spacing_y * 2) / pitch_y) + 1

        # Calculate the total size of the array to center it properly
        array_width = (nx - 1) * pitch_x
        array_height = (ny - 1) * pitch_y
        
        start_x = -array_width / 2
        start_y = -array_height / 2

        # Place holes in a loop
        for i in range(nx):
            for j in range(ny):
                c.add_ref(hole).dmove((start_x + i * pitch_x, start_y + j * pitch_y))
        
        return c

    def build(self) -> Tuple[gf.Component, int]:
        """Builds the standard mass component."""
        base_mass = self._create_base_mass()
        release_holes = self._create_release_holes()

        mass_with_holes = gf.boolean(
            A=base_mass, B=release_holes, operation="not", layer=self.layer
        )

        self.spring_spacing = (self.mass_height - 2 * self.spring_height) / (self.num_beams - 1)
        for i in range(self.num_beams):
            y_pos = -self.mass_height / 2 + self.spring_height + self.spring_spacing * i
            mass_with_holes.add_port(
                name=f"spring_r_{i}",
                center=(-self.mass_width / 2, y_pos),
                width=self.port_width,
                orientation=180,
                layer=self.layer,
            )
            mass_with_holes.add_port(
                name=f"spring_l_{i}",
                center=(self.mass_width / 2, y_pos),
                width=self.port_width,
                orientation=0,
                layer=self.layer,
            )

        mass_with_holes.add_port(
            name="photonic_crystal_attach",
            center=(0, self.mass_height / 2 + self.outcrop_height),
            width=1,
            orientation=90,
            layer=self.layer,
        )
        return mass_with_holes

    def build_inverted(self) -> Tuple[gf.Component, int]:
        """
        Builds the inverted mass component using a boolean XOR operation.
        This method is now corrected to handle locked cells.
        """
        standard_mass, num_beams = self.build()

        # Create a new component to hold the correctly positioned bounding box.
        bounding_box_comp = gf.Component()
        bbox_size = (
            standard_mass.dxsize + 2 * self.mass_padding,
            standard_mass.dysize + 2 * self.mass_padding,
        )
        # Get the primitive, locked rectangle component
        bbox_primitive = gf.components.rectangle(
            size=bbox_size, layer=self.layer, centered=True
        )
        # Add a REFERENCE to the primitive and move the REFERENCE
        bbox_ref = bounding_box_comp.add_ref(bbox_primitive)
        bbox_ref.dmove(standard_mass.dcenter)

        # Now perform the boolean operation with the correctly placed bounding box component
        inverted_mass = gf.boolean(
            A=bounding_box_comp, B=standard_mass, operation="xor", layer=self.layer
        )

        # Copy all ports from the original mass to the inverted component
        for port in standard_mass.ports:
            inverted_mass.add_port(port=port)

        return inverted_mass, num_beams


def cutback_bend180_port(
    component_bend: ComponentSpec = "bend_circular",
    component_port: ComponentSpec = "anchor_arc",
    straight: ComponentSpec = "straight",
    straight_length: float = 30.0,
    straight_vertical_length: float = 5.0,
    bend_radius: float = 3,
    anchor_radius: float = 6.0,
    rows: int = 6,
    width: float = 0.3,
    resolution: float = 0.02,
    layer: Any = (1, 0)
) -> gf.Component:
    """
    Generates a cutback structure using the custom 'anchor_arc' component.
    """

    n180 = int(2*np.pi*bend_radius/resolution) * 180/360
    n90 = int(2*np.pi*bend_radius/resolution) * 90/360

    bend180 = gf.get_component(component_bend, radius=bend_radius, width=width, angle=180, npoints=n180, allow_min_radius_violation=True, layer=layer)
    bend90 = gf.get_component(component_bend, radius=bend_radius, width=width, angle=90, npoints=n90, allow_min_radius_violation=True, layer=layer)
    straightx = gf.get_component(straight, length=straight_length, width=width, layer=layer)
    straightx_half = gf.get_component(straight, length=straight_length/ 2-bend_radius, width=width, layer=layer)
    wg_vertical = gf.get_component(straight, length=straight_vertical_length, width=width, layer=layer)
    anchor_arc_comp = gf.get_component(component_port, radius=anchor_radius, width=width, resolution=resolution, layer=layer)
    
    symbol_to_component = {
        "H": (bend90, "o1", "o2"), "K": (bend90, "o2", "o1"),
        "D": (bend180, "o1", "o2"), "C": (bend180, "o2", "o1"),
        "-": (straightx, "o1", "o2"), "<": (straightx_half, "o1", "o2"),
        ">": (straightx_half, "o1", "o2"), "|": (wg_vertical, "o1", "o2"),
        "A": (anchor_arc_comp, "o1", "o2"), "V": (anchor_arc_comp, "o2", "o1"),
    }

    if rows == 0:
        s = "V|HK|A"
    else:
        s = "V|H>"+"C-D-"*(rows-1)+"C-D<K|A"

    c = component_sequence(
        sequence=s, symbol_to_component=symbol_to_component, start_orientation=0
    )
    return c


class SerpentineSpring:
    """
    A class to generate a standard or inverted serpentine spring by wrapping
    a high-level cutback component function.
    """

    def __init__(
        self,
        beam_width: float = 0.3,
        bend_radius: float = 3.0,
        straight_length: float = 30.0,
        straight_vertical_length: float = 5.0,
        anchor_radius: float = 6.0,
        rows: int = 6,
        spring_padding: float = 2.0,
        resolution: float = 0.02,
        layer: Any = (1,0),
    ):

        self.beam_width = beam_width
        self.bend_radius = bend_radius
        self.straight_length = straight_length
        self.straight_vertical_length = straight_vertical_length
        self.anchor_radius = anchor_radius
        self.rows = rows
        self.spring_padding = spring_padding
        self.resolution = resolution
        self.layer = layer

    def build(self, **kwargs) -> gf.Component:

        spring_comp = cutback_bend180_port(
            width=self.beam_width,
            bend_radius=self.bend_radius,
            straight_length=self.straight_length,
            straight_vertical_length=self.straight_vertical_length,
            anchor_radius=self.anchor_radius,
            rows=self.rows,
            resolution=self.resolution,
            layer=self.layer,
            **kwargs
        )

        return spring_comp

    def build_inverted(self, **kwargs) -> gf.Component:

        standard_spring = self.build(**kwargs)

        bounding_box_comp = gf.Component()
        size = (standard_spring.dxsize + 2 * self.spring_padding, standard_spring.dysize)
        center = standard_spring.dcenter

        bbox_ref = bounding_box_comp.add_ref(gf.components.rectangle(size=size, layer=self.layer, centered=True))
        bbox_ref.dmove(center)
        
        inverted_spring = gf.boolean(A=bounding_box_comp, B=standard_spring, operation="not", layer=self.layer)
        
        if "o1" in standard_spring.ports and "o2" in standard_spring.ports:
            inverted_spring.add_port(name="o1", port=standard_spring.ports["o1"])
            inverted_spring.add_port(name="o2", port=standard_spring.ports["o2"])
            
        return inverted_spring
    

@register_device("optomechanical_mass")
def optomechanical_mass(
    mass_width: float = 120,
    mass_height: float = 150,
    outcrop_width: float = 20,
    outcrop_height: float = 5,
    release_hole_width: float = 2,
    release_hole_spacing_x: float = 2,
    release_hole_height: float = 6,
    release_hole_spacing_y: float = 2.75,
    spring_height: float = 10,
    num_beams: int = 3,

    beam_width: float = 0.3,
    bend_radius: float = 3.0,
    straight_length: float = 30.0,
    straight_vertical_length: float = 5.0,
    anchor_radius: float = 6.0,
    rows: int = 6,

    mass_padding: float = 2.0,
    resolution: float = 0.02,
    layer: Any = (1,0),
    **kwargs
) -> gf.Component:
    """
    Create an optomechanical mass component.
    """
    c = gf.Component()

    port_width = beam_width + 2* anchor_radius
    mass = OptoMechanicalMass(
        mass_width=mass_width,
        mass_height=mass_height,
        outcrop_width=outcrop_width,
        outcrop_height=outcrop_height,
        release_hole_width=release_hole_width,
        release_hole_spacing_x=release_hole_spacing_x,
        release_hole_height=release_hole_height,
        release_hole_spacing_y=release_hole_spacing_y,
        spring_height=spring_height,
        port_width=port_width,
        num_beams=num_beams,
        layer=layer
    )
    mass_ref = c.add_ref(mass.build())

    spring = SerpentineSpring(
        beam_width=beam_width,
        bend_radius=bend_radius,
        straight_length=straight_length,
        straight_vertical_length=straight_vertical_length,
        anchor_radius=anchor_radius,
        rows=rows,
        resolution=resolution,
        layer=layer
    )
    spring_instance = spring.build(**kwargs)

    for i in range(num_beams):
        spring_ref_r = c.add_ref(spring_instance)
        spring_ref_r.connect('o1', mass_ref.ports[f"spring_r_{i}"])
        spring_ref_l = c.add_ref(spring_instance)
        spring_ref_l.connect('o2', mass_ref.ports[f"spring_l_{i}"])

    δy = spring_height - bend_radius-beam_width/2-straight_length/2 - mass_padding
    y0 = straight_length + 2*bend_radius + beam_width + 2 * mass_padding
    x0 = 4 * rows * bend_radius + 2 * (anchor_radius + straight_vertical_length) + mass_padding

    padding_start_point = [mass_width/2+mass_padding, 0]

    padding_middle_points_nested = [[[mass_width/2+mass_padding, -outcrop_height-δy-i * mass.spring_spacing],
    [mass_width/2+x0, -outcrop_height-δy-i * mass.spring_spacing],
    [mass_width/2+x0, -outcrop_height-δy-y0-i * mass.spring_spacing],
    [mass_width/2+mass_padding, -outcrop_height-δy-y0-i * mass.spring_spacing]] for i in range(num_beams)]

    padding_middle_points = [point for group in padding_middle_points_nested for point in group]

    padding_end_point = [mass_width/2+mass_padding, -outcrop_height-mass_height-mass_padding]
    padding_points = np.array([padding_start_point] + padding_middle_points + [padding_end_point])

    padding_points_flip = padding_points.copy()
    padding_points_flip[:, 0] *= -1
    padding_points_flip = np.flip(padding_points_flip, axis=0)
    polygon_points = np.concatenate((padding_points, padding_points_flip), axis=0)

    padding_polygon = gf.Component()
    padding_polygon.add_polygon(polygon_points, layer=layer)

    padding_polygon.ymax = c.ymax
    inverted_c = gf.boolean(A=padding_polygon, B=c, operation="not", layer=layer)

    inverted_c.add_port(
            name="top",
            center=(0, inverted_c.ymax),
            width=1,
            orientation=90,
            layer=layer,
        )
    inverted_c.add_port(name="coupler", center=(0, 0), width=1, orientation=0, layer=layer)
    return inverted_c, None
