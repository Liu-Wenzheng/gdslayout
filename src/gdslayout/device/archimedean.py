import numpy as np
from ..utils import bend8_initial_guess_iter, path_to_polygon, translate_array
from gdsfactory.typings import Any
import gdsfactory as gf

class ArchimedeanSpiral:
    """
    r(θ_rel) = r0 + a*θ_rel, with absolute angle θ_abs = θ0 + θ_rel.
    All methods take θ as the RELATIVE sweep from θ0, so arc_length(0)=0.
    Radians I/O. Scalar-in ⇒ scalar-out. Arrays preserved.
    """
    def __init__(self, r0: float, a: float, theta0: float = 0.0):
        self.r0, self.a, self.theta0 = float(r0), float(a), float(theta0)

    # -------- helpers --------
    def _asarr(self, th):
        return np.asarray(th, dtype=float), np.isscalar(th)
    def _ret(self, v, is_scalar):
        v = np.asarray(v); return float(v) if is_scalar else v
    @staticmethod
    def _wrap_pi(x):
        return (x + np.pi) % (2*np.pi) - np.pi
    def _abs(self, th):
        return th + self.theta0
    def _xy_from_theta(self, theta_array):
        th = np.asarray(theta_array)
        r = self.r0 + self.a*th
        th_abs = self.theta0 + th
        x, y = r*np.cos(th_abs), r*np.sin(th_abs)
        return np.column_stack([x, y])

    # -------- core geometry --------
    def r(self, theta):
        th, s = self._asarr(theta)
        return self._ret(self.r0 + self.a*th, s)

    def xy(self, theta):
        th, s = self._asarr(theta)
        r = self.r(th)
        th_abs = self._abs(th)
        x, y = r*np.cos(th_abs), r*np.sin(th_abs)
        return (float(x), float(y)) if s else (x, y)

    def ds_dtheta(self, theta):
        th, s = self._asarr(theta)
        r = self.r(th)
        return self._ret(np.sqrt(r*r + self.a*self.a), s)

    def arc_length(self, theta, theta_start: float = 0.0):
        th, s = self._asarr(theta)
        t0 = float(theta_start)
        if self.a == 0.0:
            return self._ret(abs(self.r0)*(th - t0), s)
        a = self.a
        u  = self.r0 + a*th
        u0 = self.r0 + a*t0
        F  = lambda w: 0.5/a * (w*np.sqrt(w*w + a*a) + a*a*np.arcsinh(w/a))
        return self._ret(F(u) - F(u0), s)

    def tangent_angle(self, theta, unwrap: bool = False):
        th, s = self._asarr(theta)
        r = self.r(th)
        psi = self._abs(th) + np.arctan2(r, self.a)
        if unwrap and not s:
            psi = np.unwrap(psi)
        else:
            psi = self._wrap_pi(psi)
        return self._ret(psi, s)

    def curvature(self, theta):
        th, s = self._asarr(theta)
        r = self.r(th); a2 = self.a*self.a
        return self._ret((r*r + 2*a2) / (r*r + a2)**1.5, s)

    def dcurvature_dtheta(self, theta):
        th, s = self._asarr(theta)
        r, a = self.r(th), self.a
        return self._ret(-a*r*(r*r + 4*a*a) / (r*r + a*a)**2.5, s)

    def dcurvature_ds(self, theta):
        th, s = self._asarr(theta)
        r, a = self.r(th), self.a
        return self._ret(-a*r*(r*r + 4*a*a) / (r*r + a*a)**3, s)

    def _theta_from_s(self, svals, theta_upper, total_length=None, max_iter=25, tol=1e-12):
        svals = np.asarray(svals, dtype=float)
        if total_length is None:
            total_length = float(self.arc_length(theta_upper))
        th = (svals / max(total_length, np.finfo(float).eps)) * float(theta_upper)
        if self.a == 0.0:
            return np.clip(svals / max(abs(self.r0), np.finfo(float).eps), 0.0, float(theta_upper))
        for _ in range(max_iter):
            f  = self.arc_length(th) - svals
            fp = self.ds_dtheta(th)
            step = f / np.maximum(fp, np.finfo(float).eps)
            th_new = np.clip(th - step, 0.0, float(theta_upper))
            if np.max(np.abs(th_new - th)) < tol:
                th = th_new; break
            th = th_new
        return th

    def points(self, theta_end, resolution):
        theta_end = float(theta_end)
        L = float(self.arc_length(theta_end))
        if L == 0.0:
            return self._xy_from_theta(np.array([0.0]))

        n_seg = int(np.ceil(L / float(resolution)))  # number of segments
        n_seg = max(1, n_seg)
        s_new = np.linspace(0.0, L, n_seg + 1)       # equidistant in arc length
        th = self._theta_from_s(s_new, theta_end, total_length=L)
        return self._xy_from_theta(th)


def spiral_archimedean_inner_connector(spiral, steps=20, resolution=1):
    solver = bend8_initial_guess_iter(0, -2 * spiral.r0, 
                                      θ0=-(np.pi - spiral.tangent_angle(0)), k0=-spiral.curvature(0), k0p=-spiral.dcurvature_ds(0), 
                                      θ1=-(np.pi - spiral.tangent_angle(0)), k1=spiral.curvature(0), k1p=spiral.dcurvature_ds(0), 
                                      steps=steps)
    x_pos, y_pos = solver.draw_path(plot=False)
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    dx = np.diff(x_pos)
    dy = np.diff(y_pos)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)

    length = np.sum(ds)
    n = int(length/resolution)

    x_pos, y_pos = solver.draw_path(n, plot=False)
    return np.array(x_pos), np.array(y_pos)


def spiral_archimedean_outer_coupling_connector(spiral, theta1, theta2, transit_length=120, transit_depth=5, flat_length = 40, boundary_length=None, boundary_depth=None, boundary_radius=None, steps=10, resolution = 1, output=False):

    
    if boundary_length is not None and boundary_depth is not None and boundary_radius is not None:
        bounding = True

        solver = bend8_initial_guess_iter(l=boundary_length, h=boundary_depth,
                                      θ0=-(np.pi - spiral.tangent_angle(theta1)), k0=spiral.curvature(theta1), k0p=spiral.dcurvature_ds(theta1),
                                      θ1=0, k1=-1/boundary_radius, k1p=0,
                                      steps=steps, output=output)
        
        transit_boundary = bend8_initial_guess_iter(l=-(boundary_length-transit_length - flat_length), h=-(boundary_depth+spiral.a*np.pi-transit_depth),
                                      θ0=0, k0=-1/boundary_radius, k0p=0,
                                      θ1=-np.pi, k1=0, k1p=0,
                                      steps=steps, output=output)
    
        x_pos_b, y_pos_b = transit_boundary.draw_path(plot=False)

        dx = np.diff(x_pos_b)
        dy = np.diff(y_pos_b)
        ds = np.sqrt((dx) ** 2 + (dy) ** 2)
        length = np.sum(ds)
        n = int(length/resolution)
        x_pos_b, y_pos_b = transit_boundary.draw_path(n, plot=False)
    else:
        bounding = False
        solver = bend8_initial_guess_iter(l=flat_length+transit_length, h=-spiral.a * np.pi + transit_depth,
                                      θ0=-(np.pi - spiral.tangent_angle(theta1)), k0=spiral.curvature(theta1), k0p=spiral.dcurvature_ds(theta1),
                                      θ1=-np.pi, k1=0, k1p=0,
                                      steps=steps)


    x_pos, y_pos = solver.draw_path(plot=False)

    dx = np.diff(x_pos)
    dy = np.diff(y_pos)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)
    length = np.sum(ds)
    n = int(length/resolution)
    x_pos, y_pos = solver.draw_path(n, plot=False)


    transit = bend8_initial_guess_iter(l=transit_length, h=-transit_depth,
                                      θ0=-spiral.tangent_angle(theta2), k0=-spiral.curvature(theta2), k0p=spiral.dcurvature_ds(theta2),
                                      θ1=0, k1=0, k1p=0,
                                      steps=steps, output=output)
    
    x_pos_t, y_pos_t = transit.draw_path(plot=False)

    dx = np.diff(x_pos_t)
    dy = np.diff(y_pos_t)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)
    length = np.sum(ds)
    n = int(length/resolution)
    x_pos_t, y_pos_t = transit.draw_path(n, plot=False)

    if bounding:
        n_straight = int(x_pos_b[0]/resolution)
        x_straight = np.linspace(-x_pos_t[-1], -x_pos_b[0], n_straight)
        y_straight = np.linspace(y_pos_t[-1]+spiral.a* np.pi, -y_pos_b[0], n_straight)

        x_combined = np.concatenate((-x_pos_t, x_straight[1:-1], -x_pos_b[::-1][:-1]-boundary_length, -x_pos[::-1]))
        y_combined = np.concatenate((y_pos_t+spiral.a* np.pi , y_straight[1:-1], -y_pos_b[::-1][:-1]-boundary_depth, -y_pos[::-1]))
    else:
        n_straight = int(x_pos[-1]/resolution)
        x_straight = np.linspace(-x_pos_t[-1], -x_pos[-1], n_straight)
        y_straight = np.linspace(y_pos_t[-1]+spiral.a* np.pi, -y_pos[-1], n_straight)

        x_combined = np.concatenate((-x_pos_t, x_straight[1:-1], -x_pos[::-1]))
        y_combined = np.concatenate((y_pos_t+spiral.a* np.pi , y_straight[1:-1], -y_pos[::-1]))

    return x_combined, y_combined, len(x_pos_t)


def spiral_archimedean_outer_connector(spiral, theta1, theta2, boundary_length=250, boundary_depth=50, boundary_radius=200, steps=10, resolution=1, output=False):

    solver = bend8_initial_guess_iter(l=boundary_length, h=boundary_depth,
                                      θ0=-(np.pi - spiral.tangent_angle(theta1)), k0=spiral.curvature(theta1), k0p=spiral.dcurvature_ds(theta1),
                                      θ1=0, k1=-1/boundary_radius, k1p=0,
                                      steps=steps, output=output)
    
    x_pos, y_pos = solver.draw_path(plot=False)

    dx = np.diff(x_pos)
    dy = np.diff(y_pos)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)
    length = np.sum(ds)
    n = int(length/resolution)
    x_pos, y_pos = solver.draw_path(n, plot=False)
    transit = bend8_initial_guess_iter(l=boundary_length, h=-boundary_depth-spiral.a * np.pi,
                                      θ0=-spiral.tangent_angle(theta2), k0=-spiral.curvature(theta2), k0p=spiral.dcurvature_ds(theta2),
                                      θ1=-np.pi, k1=-1/boundary_radius, k1p=0,
                                      steps=steps, output=output)
    
    x_pos_t, y_pos_t = transit.draw_path(plot=False)

    dx = np.diff(x_pos_t)
    dy = np.diff(y_pos_t)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)
    length = np.sum(ds)
    n = int(length/resolution)
    x_pos_t, y_pos_t = transit.draw_path(n, plot=False)

    x_combined = np.concatenate((-x_pos_t, -x_pos[::-1][1:]))
    y_combined = np.concatenate((y_pos_t+spiral.a* np.pi , -y_pos[::-1][1:]))

    return x_combined, y_combined


def spiral_archimedean1(
        flat_length: float = 300,
        transit_length: float = 120,
        transit_depth: float = 5,
        boundary_length: Any = None,
        boundary_depth: Any = None,
        boundary_radius: Any = None,
        width: float = 1, 
        num_turns: int = 5, 
        spiral_gap: float = 5, 
        inner_gap: float = 10,
        translation: float = 0, 
        resolution: float = 1, 
        layer: Any = (1, 0)
    ):

    """
    Returns a spiral component.
    """

    r0 = inner_gap/2
    a = spiral_gap/np.pi

    theta_start = np.pi/2
    theta1 = num_turns * 2 * np.pi
    theta2 = (num_turns + 0.5) * 2 * np.pi

    spiral = ArchimedeanSpiral(r0=r0, a=a, theta0=theta_start)
    points1 = spiral.points(theta1, resolution)
    points2 = - spiral.points(theta2, resolution)

    x_arr , y_arr = spiral_archimedean_inner_connector(spiral, resolution=resolution)
    point_mid = np.column_stack((x_arr , y_arr + inner_gap/2))[::-1]

    x_arr , y_arr, flat_point_start = spiral_archimedean_outer_coupling_connector(spiral, theta1, theta2, flat_length=flat_length, transit_length=transit_length, transit_depth=transit_depth, 
                                                                boundary_length=boundary_length, boundary_depth=boundary_depth, boundary_radius=boundary_radius, steps=800, resolution=resolution)
    point_init = np.column_stack((x_arr , y_arr + spiral.r(theta1)))[::-1]

    points = np.concatenate([np.array(points2)[::-1], point_mid[1:-1], np.array(points1), point_init[1:]], axis=0)
    points = translate_array(points, translation)
    path = gf.Path(points)
    x_c, y_c = point_init[-flat_point_start-1]

    component = path_to_polygon(path, width_fn=width, layer=layer)[0]
    component.add_port(name="coupler", center=(x_c, y_c+width/2), width=1, orientation=0, layer=layer)
    return component, path


def spiral_archimedean2(
        boundary_length: float = 150,
        boundary_depth: float = 68,
        boundary_radius: float = 35,
        width: float = 1, 
        num_turns: int = 5, 
        spiral_gap: float = 5, 
        inner_gap: float = 10,
        translation: float = 0, 
        resolution: float = 1, 
        layer: Any = (1, 0)
    ):

    """
    Returns a spiral component.
    """

    r0 = inner_gap/2
    a = spiral_gap/np.pi

    theta_start = np.pi/2
    theta1 = num_turns * 2 * np.pi
    theta2 = (num_turns + 0.5) * 2 * np.pi

    spiral = ArchimedeanSpiral(r0=r0, a=a, theta0=theta_start)
    points1 = spiral.points(theta1, resolution)
    points2 = - spiral.points(theta2, resolution)

    x_arr , y_arr = spiral_archimedean_inner_connector(spiral, resolution=resolution)
    point_mid = np.column_stack((x_arr , y_arr + inner_gap/2))[::-1]

    x_arr , y_arr = spiral_archimedean_outer_connector(spiral, theta1, theta2, boundary_length=boundary_length, boundary_depth=boundary_depth, boundary_radius=boundary_radius, steps=800, resolution=resolution)
    point_init = np.column_stack((x_arr , y_arr + spiral.r(theta1)))[::-1]

    points = np.concatenate([np.array(points2)[::-1], point_mid[1:-1], np.array(points1), point_init[1:]], axis=0)
    points = translate_array(points, translation)
    path = gf.Path(points)
    x_c, y_c = points1[-1]

    component = path_to_polygon(path, width_fn=width, layer=layer)[0]
    component.add_port(name="coupler", center=(x_c, y_c+width/2), width=1, orientation=0, layer=layer)
    return component, path