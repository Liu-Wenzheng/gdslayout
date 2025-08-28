import numpy as np

import gdsfactory as gf
from gdsfactory.typings import Any

from ..utils import path_to_polygon, translate_array, correct_euler
from ..utils import spiral_inner_connector, spiral_outer_connector
from . import register_device


@register_device("ring")
def ring(
    width: float = 1.0, 
    radius: float = 50.0, 
    resolution: float = 1.0, 
    layer: Any = (1, 0)
):
    """
    Returns a ring component.
    """

    n = int(2*np.pi*radius/resolution)
    angle_resolution = 360 / n
    
    D = gf.Component()
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    n = int(np.round(360 / angle_resolution))
    t = np.linspace(0, 360, n + 1) * np.pi / 180
    points_x = radius * np.cos(t)
    points_y = radius * np.sin(t)
    inner_points_x = inner_radius * np.cos(t)
    inner_points_y = inner_radius * np.sin(t)
    outer_points_x = outer_radius * np.cos(t)
    outer_points_y = outer_radius * np.sin(t)
    xpts = np.concatenate([inner_points_x, outer_points_x[::-1]])
    ypts = np.concatenate([inner_points_y, outer_points_y[::-1]])
    D.add_polygon(points=list(zip(xpts, ypts)), layer=layer)
    P = gf.Path(list(zip(points_x, points_y))).rotate(90)
    D.add_port(name="coupler", center=(0, radius+width/2), width=1, orientation=0, layer=layer)
    return D, P


@register_device("racetrack")
def race_track(
        width: float = 1.0, 
        radius: float = 50.0, 
        length: float = 200.0, 
        p: float = 1.0, 
        translation: float = 0.0, 
        resolution: float = 1.0, 
        layer: Any = (1, 0)
):
    """ 
    Returns a race track component.
    """

    turn = correct_euler(radius=radius, angle=180, p=p, resolution=resolution)

    n_straight = int(length / resolution)
    if n_straight < 2:
        n_straight = 2

    straight = gf.path.straight(length=length, npoints=n_straight)

    p1 = turn.rotate(180).points
    p2 = straight.points + p1[-1]
    p3 = turn.rotate(180).points + p2[-1]
    p4 = straight.rotate(180).points + p3[-1]
    points = np.concatenate((p1, p2[1:-1], p3, p4[1:-1]))

    points = translate_array(points, translation)
    points[:, 0] -= length / 2
    points[:, 1] += radius
    path = gf.Path(points)
    x_c, y_c = points[0]

    component = path_to_polygon(path, width_fn=width, layer=layer)[0]
    component.add_port(name="coupler", center=(x_c, y_c+width/2), width=1, orientation=0, layer=layer)
    return component, path


@register_device("spiral")
def spiral(
        straight_length: float = 50, 
        left_length: float = 300, 
        right_length: float = 150, 
        width: float = 1, 
        num_turns: int = 5, 
        spiral_gap: float = 5, 
        inner_gap: float = 10, 
        p: float = 1, 
        translation: float = 0, 
        resolution: float = 1, 
        layer: Any = (1, 0)
    ):
    """
    Returns a spiral component.
    """

    n_straight = int(straight_length / resolution)
    if n_straight < 2:
        n_straight = 2
    
    P1 = gf.Path()
    radius = (np.arange(num_turns) * 2*spiral_gap + inner_gap + spiral_gap) / 2
    
    for r in radius:
        turn = correct_euler(radius=r, angle=180, p=p, resolution=resolution, use_eff=True)
        straight = gf.path.straight(length=straight_length, npoints=n_straight)
        P1 += turn + straight

    P2 = P1.copy()
    if num_turns % 2 == 0:
        P1.rotate(180)
        P1.move((-straight_length, inner_gap))
    else:
        P2.rotate(180)
        P2.move((-straight_length, inner_gap))

    P2 += correct_euler(radius=r+spiral_gap, angle=180, p=p, resolution=resolution, use_eff=True)

    P2 +=  gf.path.straight(length=straight_length, npoints=n_straight)
    points1 = P1.points
    points2 = P2.points
    
    x_arr, y_arr = spiral_inner_connector(straight_length, inner_gap, resolution=resolution, output=False)
    point_mid = np.column_stack((x_arr - straight_length /2 , -y_arr + inner_gap/2))[::-1]
    
    x_arr, y_arr = spiral_outer_connector(left_length=left_length, right_length=right_length, gap=spiral_gap, steps=20, resolution=resolution, output=False)
    point_init = np.column_stack((x_arr - straight_length-right_length , y_arr + inner_gap + spiral_gap*num_turns))[::-1]
    
    # Combine all points
    if num_turns % 2 == 0:
        points = np.concatenate([np.array(points2)[::-1], point_mid[1:-1], np.array(points1), point_init[1:]], axis=0)
    else:
        points = np.concatenate([np.array(points2)[::-1], point_mid[1:-1][::-1], np.array(points1), point_init[1:]], axis=0)

    points = points[::-1]
    points = translate_array(points, translation)
    points[:, 0] += straight_length / 2
    points[:, 1] -= inner_gap /2

    path = gf.Path(points)
    x_c, y_c = points[0]

    component = path_to_polygon(path, width_fn=width, layer=layer)[0]
    component.add_port(name="coupler", center=(x_c, y_c+width/2), width=1, orientation=0, layer=layer)
    return component, path