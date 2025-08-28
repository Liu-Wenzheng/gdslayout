import numpy as np 

import gdsfactory as gf
from gdsfactory.typings import Any

from ..utils import spiral_inner_connector, pulley_coupler_connector
from . import register_coupler


@register_coupler("point")
def point_coupler(
        width: float = 1.0, 
        length: float = 100.0, 
        center: tuple = (0, 50), 
        ring_down: tuple = None, 
        resolution: float = 1.0, 
        layer: Any = (1,0)
    ):

    """
    Returns a point coupler component.
    """

    if not isinstance(length, (tuple, list)):
        length = (length, length)
    n1, n2 = int(length[0] / resolution), int(length[1] / resolution)
    x_pos = np.concatenate((np.linspace(-length[0], 0, n1)[:-1], np.linspace(0, length[1], n2)))
    y_pos = np.zeros_like(x_pos) + center[1]
    points = np.column_stack((x_pos, y_pos))
    path = gf.Path(points)
    if ring_down is not None:
        [l, d] = [ring_down[0], ring_down[1] + center[1]]
        # l, d = ring_down[0] * length[0]/2, (ring_down[1]+1) * center[1]
        x_arr, y_arr = spiral_inner_connector(length=l, gap=d, steps=40, resolution=resolution, output=False)
        points_bender = np.column_stack((x_arr, -y_arr))
        path_bender = gf.Path(points_bender)
        path.append(path_bender)

    start_point = path.points[0]
    end_point = path.points[-1]

    start_dir = path.points[0] - path.points[1]
    end_dir = path.points[-1] - path.points[-2]
    start_angle = np.arctan2(start_dir[1], start_dir[0]) * 180 / np.pi
    end_angle = np.arctan2(end_dir[1], end_dir[0]) * 180 / np.pi

    cross_section = gf.cross_section.strip(width=width, layer=layer)
    component = gf.path.extrude(path, cross_section)

    component.add_port(name='in', center=start_point, width=width, orientation=start_angle, layer=layer)
    component.add_port(name='out', center=end_point, width=width, orientation=end_angle, layer=layer)
    return component, path


@register_coupler("pulley")
def symmetric_pulley_coupler(
    width: float = 1.0, 
    length: float = 100.0, 
    height: float = 20.0, 
    theta: float = np.pi/6, 
    Rc0: float = None, 
    center: tuple = (0, 50), 
    ring_down: tuple = None, 
    resolution: float = 1.0, 
    layer: Any = (1,0)
):
    """
    Returns a symmetric pulley coupler component.
    """

    Rc = center[1]
    d = Rc - height
    if ring_down is not None:
        [l2, d2] = [ring_down[0], -ring_down[1]]
    else:
        l2, d2 = length, d
    x_arr, y_arr = pulley_coupler_connector(l1=length, l2=l2, d1=d, d2=d2, θ1=theta, θ2=theta, Rc=Rc, Rc0=Rc0, steps=40, resolution=resolution, output=False)
    points = np.column_stack((x_arr, y_arr))
    path = gf.Path(points)

    cross_section = gf.cross_section.strip(width=width, layer=layer)
    component = gf.path.extrude(path, cross_section)
    
    start_point = path.points[0]
    end_point = path.points[-1]

    start_dir = path.points[0] - path.points[1]
    end_dir = path.points[-1] - path.points[-2]
    start_angle = np.arctan2(start_dir[1], start_dir[0]) * 180 / np.pi
    end_angle = np.arctan2(end_dir[1], end_dir[0]) * 180 / np.pi

    component.add_port(name='in', center=start_point, width=width, orientation=start_angle, layer=layer)
    component.add_port(name='out', center=end_point, width=width, orientation=end_angle, layer=layer)

    return component, path

@register_coupler("asym_pulley")
def asymmetric_pulley_coupler(
    width: float = 1.0, 
    length: float = 100.0, 
    height: float = 20.0, 
    theta_in: float = np.pi/6, 
    theta_out: float = np.pi/6, 
    Rc0: float = None, 
    center: tuple = (0, 50), 
    ring_down: tuple = None, 
    resolution: float = 1.0, 
    layer: Any = (1,0)
):
    """
    Returns a symmetric pulley coupler component.
    """

    Rc = center[1]
    d = Rc - height
    if ring_down is not None:
        [l2, d2] = [ring_down[0], -ring_down[1]]
    else:
        l2, d2 = length, d
    x_arr, y_arr = pulley_coupler_connector(l1=length, l2=l2, d1=d, d2=d2, θ1=theta_in, θ2=theta_out, Rc=Rc, Rc0=Rc0, steps=40, resolution=resolution, output=False)
    points = np.column_stack((x_arr, y_arr))
    path = gf.Path(points)

    cross_section = gf.cross_section.strip(width=width, layer=layer)
    component = gf.path.extrude(path, cross_section)
    
    start_point = path.points[0]
    end_point = path.points[-1]

    start_dir = path.points[0] - path.points[1]
    end_dir = path.points[-1] - path.points[-2]
    start_angle = np.arctan2(start_dir[1], start_dir[0]) * 180 / np.pi
    end_angle = np.arctan2(end_dir[1], end_dir[0]) * 180 / np.pi

    component.add_port(name='in', center=start_point, width=width, orientation=start_angle, layer=layer)
    component.add_port(name='out', center=end_point, width=width, orientation=end_angle, layer=layer)

    return component, path

@register_coupler("out_point")
def out_point_coupler(
    width: float = 1.0, 
    length: float = 50.0, 
    height: float = 20.0, 
    coupling_length: float = 50.0, 
    Rc: float = 50.0, 
    Rc0: float = None, 
    extend_length: float = 100.0, 
    center: tuple = (0, 50), 
    ring_down: tuple = None, 
    resolution: float = 1.0, 
    layer: Any = (1,0)
):
    """
    Returns an out-point coupler component.
    """

    if coupling_length:
        x_pos_in, y_pos_in = spiral_inner_connector(length=length, gap=height, steps=40, resolution=resolution, output=False)
        x_pos_c = np.linspace(-coupling_length/2, coupling_length/2, int(coupling_length/resolution))
        x_arr = np.concatenate((x_pos_in-length/2-coupling_length/2, x_pos_c[1:-1], -x_pos_in[::-1]+length/2+coupling_length/2))
        y_arr = np.concatenate((-y_pos_in-height/2, np.zeros_like(x_pos_c)[1:-1]-height, -y_pos_in[::-1]-height/2))
    elif coupling_length ==0 and Rc0 is not None:
        x_arr, y_arr = pulley_coupler_connector(l1=length, l2=length, d1=Rc-height, d2=Rc-height, θ1=0, θ2=0, Rc=Rc, Rc0=Rc0, steps=40, resolution=resolution, output=False)
        y_arr = -y_arr+Rc-height
    else:
        x_arr, y_arr = pulley_coupler_connector(l1=length, l2=length, d1=Rc-height, d2=Rc-height, θ1=np.pi/6, θ2=np.pi/6, Rc=Rc, steps=40, resolution=resolution, output=False)
        y_arr = -y_arr+Rc-height

    points = np.column_stack((x_arr, y_arr + center[1] + height))
    points[:, 0] += center[0]
    path = gf.Path(points)

    if ring_down is not None:
        # l, d = ring_down[0] * length, (ring_down[1]+1) * (center[1]+height)
        l, d = ring_down[0], ring_down[1]+center[1]+height
        x_arr, y_arr = spiral_inner_connector(length=l, gap=d, steps=40, resolution=resolution, output=False)
        if extend_length:
            points_extend_straight = np.linspace(-extend_length/2, extend_length/2, int(extend_length/resolution))
            path_extend_straight = gf.Path(np.column_stack((points_extend_straight, np.zeros_like(points_extend_straight) - center[1])))
            path.append(path_extend_straight)
        points_bender = np.column_stack((x_arr, -y_arr))
        path_bender = gf.Path(points_bender)
        path.append(path_bender)
    
    cross_section = gf.cross_section.strip(width=width, layer=layer)
    component = gf.path.extrude(path, cross_section)
    
    start_point = path.points[0]
    end_point = path.points[-1]

    start_dir = path.points[0] - path.points[1]
    end_dir = path.points[-1] - path.points[-2]
    start_angle = np.arctan2(start_dir[1], start_dir[0]) * 180 / np.pi
    end_angle = np.arctan2(end_dir[1], end_dir[0]) * 180 / np.pi

    component.add_port(name='in', center=start_point, width=width, orientation=start_angle, layer=layer)
    component.add_port(name='out', center=end_point, width=width, orientation=end_angle, layer=layer)
    return component, path


@register_coupler("dual_out_point")
def dual_out_point_coupler(
    width: float = 1.0, 
    length: tuple = (50.0, 50.0), 
    height: tuple = (20.0, 20.0), 
    coupling_length: tuple = (5.0, 5.0), 
    Rc: tuple = (50.0, 50.0), 
    Rc0: tuple = (None, None), 
    extend_length: float = 100.0,
    center: tuple = (0, 50),
    center1_shift: tuple = (160, 30),
    ring_down: tuple = None, 
    resolution: float = 1.0, 
    layer: Any = (1,0)
):
    _, path1 = out_point_coupler(
        width=width, 
        length=length[0], 
        height=height[0], 
        coupling_length=coupling_length[0], 
        Rc=Rc[0], 
        Rc0=Rc0[0], 
        extend_length=extend_length, 
        center=center, 
        ring_down= None, 
        resolution=resolution, 
        layer=layer
    )
    _, path2 = out_point_coupler(
        width=width, 
        length=length[1], 
        height=height[1], 
        coupling_length=coupling_length[1], 
        Rc=Rc[1], 
        Rc0=Rc0[1], 
        extend_length=extend_length, 
        center=(center[0]+center1_shift[0], center[1]+center1_shift[1]), 
        ring_down=ring_down, 
        resolution=resolution, 
        layer=layer
    )

    points1 = path1.points
    points2 = path2.points
    x1, y1 = points1[-1]
    x2, y2 = points2[0]
    if y1 == y2:
        points = np.concatenate((points1, points2))
    else:
        x_arr, y_arr = spiral_inner_connector(length=x2-x1, gap=y2-y1, steps=40, resolution=resolution, output=False)
        points = np.concatenate((points1, np.column_stack((x_arr+(x1-x_arr[0]), y_arr+(y1-y_arr[0])))[1:], points2[1:]), axis=0)
    path = gf.Path(points)

    cross_section = gf.cross_section.strip(width=width, layer=layer)
    component = gf.path.extrude(path, cross_section)
    
    start_point = path.points[0]
    end_point = path.points[-1]

    start_dir = path.points[0] - path.points[1]
    end_dir = path.points[-1] - path.points[-2]
    start_angle = np.arctan2(start_dir[1], start_dir[0]) * 180 / np.pi
    end_angle = np.arctan2(end_dir[1], end_dir[0]) * 180 / np.pi

    component.add_port(name='in', center=start_point, width=width, orientation=start_angle, layer=layer)
    component.add_port(name='out', center=end_point, width=width, orientation=end_angle, layer=layer)
    return component, path


def _calculate_normals(points: np.ndarray) -> np.ndarray:
    """
    Calculates the normalized normal vector for each point on a path.
    The normal is defined as the 90-degree counter-clockwise rotation
    of the tangent vector.
    """

    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1.0
    unit_tangents = tangents / norms[:, np.newaxis]
    normals = np.column_stack([-unit_tangents[:, 1], unit_tangents[:, 0]])
    return normals


@register_coupler("port_coupler")
def port_coupler(
    width: float = 1.0,
    side: str = 'left',
    length: float = 100.0,
    center: tuple[float, float] = (0, 50),
    ring_down: bool | None = None,
    etch_info: tuple[float, float] | None = None,
    resolution: float = 1.0,
    layer: Any = (1, 0)
):
    """
    Returns a port coupler component with a manually generated asymmetric etch polygon.
    """

    if side == 'left':
        c, p = point_coupler(width=width, length=(length, 0), center=center, resolution=resolution, layer=layer)
    elif side == 'right':
        c, p = point_coupler(width=width, length=(0, length), ring_down=ring_down, center=center, resolution=resolution, layer=layer)
    else:
        raise ValueError("side must be 'left' or 'right'")

    if etch_info is not None:
        """if etch_info == 'b':
            width_pos = etch_width / 2 
            width_neg = center[1]
        elif etch_info == 't':
            width_pos = center[0]
            width_neg = etch_width / 2
        elif etch_info == 'a':
            width_pos = center[0]
            width_neg = center[1]
        else:
            width_pos = etch_width / 2
            width_neg = etch_width / 2"""
        width_pos, width_neg = etch_info

        points = p.points
        normals = _calculate_normals(points)
        top_points = points + normals * width_pos
        bottom_points = points - normals * width_neg

        polygon_points = np.concatenate([top_points, np.flipud(bottom_points)])
        
        etch_c = gf.Component()
        etch_c.add_polygon(polygon_points, layer=layer)

        etched_c = gf.boolean(A=etch_c, B=c, operation='not', layer=layer)
        if side == 'left':
            etched_c.add_port_with_info(name='in', center=c.ports['in'].center, width=width, orientation=c.ports['in'].orientation, layer=layer, info={'etch': True, 'etch_width_u': width_pos, 'etch_width_b': width_neg})
            etched_c.add_port(name='out', center=c.ports['out'].center, width=1, orientation=c.ports['out'].orientation, layer=layer)
        elif side == 'right':
            etched_c.add_port(name='in', center=c.ports['in'].center, width=1, orientation=c.ports['in'].orientation, layer=layer)
            etched_c.add_port_with_info(name='out', center=c.ports['out'].center, width=width, orientation=c.ports['out'].orientation, layer=layer, info={'etch': True, 'etch_width_u': width_pos, 'etch_width_b': width_neg})
        else:
            raise ValueError("side must be 'left' or 'right'")
        return etched_c, p
    else:
        return c, p