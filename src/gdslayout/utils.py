import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

import gdsfactory as gf
from numpy.typing import NDArray
from typing import Union, Callable

def translate_array(arr, percentage):
    
    if 0 < np.abs(percentage) < 1:

        x = arr[:, 0]
        y = arr[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt((dx) ** 2 + (dy) ** 2)
        s = np.cumsum(ds)
        if percentage < 0:
            s_shift = - percentage * s[-1]
        else:
            s_shift = s[-1] - percentage * s[-1]

        index = np.searchsorted(s, s_shift)

        # Rearrange the array
        arr_cut = arr[index:]
        translated_arr = np.vstack((arr_cut[:-1], arr[:index], np.array([arr_cut[0]])))

    elif np.abs(percentage) > 1:
        shift = int(percentage)
        translated_arr = np.roll(arr[1:], -shift)
        translated_arr = np.insert(translated_arr, 0, translated_arr[-1, :], axis=0)
    
    else:
        return arr.copy()

    return translated_arr

def coupler(config_edge_coupler, side=None, resolution=0.1, layer=(2, 0)):
    if side is None:
        if config_edge_coupler['file']:
            edge_coupler = gf.import_gds(config_edge_coupler['file'], cellname=config_edge_coupler['cellname'])
            edge_coupler.add_port(name="out", center=(edge_coupler.xmin, edge_coupler.y), width=config_edge_coupler['width'], orientation=180, layer=(2, 0))
            return edge_coupler, config_edge_coupler['offset'], config_edge_coupler['length'] - config_edge_coupler['offset'], config_edge_coupler['width'], config_edge_coupler['port_len']
        elif config_edge_coupler['type'] == 'straight':
            return None, 0, 0, config_edge_coupler['width'], 0
        elif config_edge_coupler['type'] == 'bending':
            edge_coupler = bending_coupler(config_edge_coupler['length'], config_edge_coupler['bending_height'], np.pi * config_edge_coupler['angle']/180, width=config_edge_coupler['width'], resolution=resolution, layer=layer)
            return edge_coupler, 0, config_edge_coupler['length']-config_edge_coupler['width']*np.sin(np.pi * config_edge_coupler['angle']/180)/2, config_edge_coupler['width'], 0
    else:
        if config_edge_coupler[side]['file']:
            edge_coupler = gf.import_gds(config_edge_coupler[side]['file'], cellname=config_edge_coupler[side]['cellname'])
            edge_coupler.add_port(name="out", center=(edge_coupler.xmin, edge_coupler.y), width=config_edge_coupler[side]['width'], orientation=180, layer=(2, 0))
            return edge_coupler, config_edge_coupler[side]['offset'], config_edge_coupler[side]['length'] - config_edge_coupler[side]['offset'], config_edge_coupler[side]['width'], config_edge_coupler[side]['port_len']
        elif config_edge_coupler[side]['type'] == 'straight':
            return None, 0, 0, config_edge_coupler[side]['width'], 0
        elif config_edge_coupler[side]['type'] == 'bending':
            edge_coupler = bending_coupler(config_edge_coupler[side]['length'], config_edge_coupler[side]['bending_height'], np.pi * config_edge_coupler[side]['angle']/180, width=config_edge_coupler[side]['width'], resolution=resolution, layer=layer)
            return edge_coupler, 0, config_edge_coupler[side]['length']-config_edge_coupler[side]['width']*np.sin(np.pi * config_edge_coupler[side]['angle']/180)/2, config_edge_coupler[side]['width'], 0

def bending_coupler(l_range, h, θ0, width, resolution, xatol=0.1, test=False, layer=(2, 0)):

    def curve(l, h, θ0, calculation=True):
        solver = bend8_initial_guess_iter(l, h, θ0=θ0, k0=0, k0p=0, θ1=0, k1=0, k1p=0, steps=100)
        x_pos_in, y_pos_in = solver.draw_path(plot=False)
        dx = np.diff(x_pos_in)
        dy = np.diff(y_pos_in)
        ds = np.sqrt((dx) ** 2 + (dy) ** 2)
        length_s = np.sum(ds)   
        n = int(length_s/resolution)
        x_pos_in, y_pos_in = solver.draw_path(n, plot=False)
        points = np.column_stack((x_pos_in, y_pos_in))
        path = gf.Path(points)
        curve = path.curvature()
        if calculation:
            return np.max(curve[1])
        else:
            return path

    if isinstance(l_range, (int, float, np.integer, np.floating)) or (isinstance(l_range, (list, np.ndarray)) and len(l_range) == 1):
        optimal_l = l_range if not isinstance(l_range, (list, np.ndarray)) else l_range[0]
        path = curve(optimal_l, h, θ0, calculation=False)
        curve_vals = path.curvature()
        component1 = gf.path.extrude(path, layer=layer, width=width)
        component2 = gf.components.rectangle(size=(width, width/np.cos(θ0)), centered=False, layer=layer)
        c = gf.Component()
        c << component1
        component2_ref = c.add_ref(component2)
        component2_ref.move((width * (np.sin(θ0)/2-1), -width * np.cos(θ0)/2))
        intersection = gf.boolean(component1, component2_ref, operation="not", layer=layer)
        c1 = gf.Component()
        c1 << intersection
        c1.add_port(name="coupler", center=(optimal_l, h), width=width, layer=layer)
        return c1.rotate(180)
    
    elif isinstance(l_range, (list, np.ndarray)) and len(l_range) == 2:
        result = minimize_scalar(curve, bounds=l_range, args=(h, θ0), options={'xatol': xatol, 'maxiter': 30}, method='bounded')
        optimal_l = result.x
        if optimal_l - xatol < l_range[0]:
            print("Warning: The optimal value is close to the upper boundary of the search range.")
        elif optimal_l + xatol > l_range[1]:
            print("Warning: The optimal value is close to the lower boundary of the search range.")
        path = curve(optimal_l, h, θ0, calculation=False)
        curve_vals = path.curvature()
        if test:
            print(f"max curvature: {np.max(curve_vals[1])}", f"min curvature: {np.min(curve_vals[1])}")
            plt.plot(curve_vals[0], curve_vals[1])
            plt.xlabel("path")
            plt.ylabel("curvature")
            plt.show()

        print(f"l_optimal = {optimal_l}")
        component1 = gf.path.extrude(path, layer=layer, width=width)
        component2 = gf.components.rectangle(size=(width, width/np.cos(θ0)), centered=False, layer=layer)
        c = gf.Component()
        c << component1
        component2_ref = c.add_ref(component2)
        component2_ref.move((width * (np.sin(θ0)/2-1), -width * np.cos(θ0)/2))
        intersection = gf.boolean(component1, component2_ref, operation="not", layer=layer)
        c1 = gf.Component()
        c1 << intersection
        c1.add_port(name="coupler", center=(optimal_l, h), width=width, layer=layer)
        return c1
    else:
        raise ValueError("l_range must be a single value or a list/array of length 2.")


def path_to_polygon(
    path: Union[gf.Path, NDArray[np.floating]],
    width_fn: Union[Callable, float, int, NDArray[np.floating]] = 0.5,
    *,
    layer: tuple[int, int] = (1, 0),
) -> tuple[gf.Component, NDArray[np.floating]]:
    """Centre‑line + width profile ➜ single watertight polygon Component.

    Parameters
    ----------
    path   : gf.Path | (N,2) ndarray
    width_fn : callable | float | int | array
         if callable, must return an array of widths the same length as `s`
         if number, a constant width is used for the entire path
         if array, must have the same length as the path points
    layer  : (layer, datatype) tuple

    Returns
    -------
    component, polygon_vertices
    """

    # 1 ── extract Nx2 centre‑line ------------------------------------------------
    xy = path.points.copy() if isinstance(path, gf.Path) else np.asarray(path, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("`path` must be Nx2 array or gf.Path with 2‑D points")

    closed = np.allclose(xy[0], xy[-1])
    if closed and len(xy) > 1:
        xy = xy[:-1]                        # drop duplicate

    # 2 ── arclength + width ------------------------------------------------------
    seg_len = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))   # segment lengths
    cum_len = np.hstack(([0.0], np.cumsum(seg_len)))           # cumulative [µm]
    s = cum_len / cum_len[-1]                           # 0 … 1
    
    # Handle constant width, width function, or width array
    if callable(width_fn):
        w = np.asarray(width_fn(s), dtype=float)
        if w.shape != s.shape:
            raise ValueError("width_fn(s) must return an array len(s)")
    elif isinstance(width_fn, (list, tuple, np.ndarray)):
        # Handle array-like width
        w = np.asarray(width_fn, dtype=float)
        if closed and len(w) > 1:
            w = w[:-1]                        # drop duplicate
        if len(w) == len(xy):
            # Width array matches path points
            pass
        elif len(w) == 1:
            # Single element array, treat as constant
            w = np.full_like(s, w[0], dtype=float)
        else:
            raise ValueError("width array must have same length as path points or be a single value")
    else:
        # Use constant width
        w = np.full_like(s, width_fn, dtype=float)

    # Ensure w is a numpy array
    w = np.asarray(w, dtype=float)
    half = 0.5 * w

    # 3 ── tangents & normals -----------------------------------------------------
    if closed:                              # periodic centred diff
        dx = np.roll(xy[:, 0], -1) - np.roll(xy[:, 0], 1)
        dy = np.roll(xy[:, 1], -1) - np.roll(xy[:, 1], 1)
    else:                                   # open path
        dx = np.gradient(xy[:, 0])
        dy = np.gradient(xy[:, 1])
        dx[0], dy[0]   = xy[1]  - xy[0]      # use first & last segments
        dx[-1], dy[-1] = xy[-1] - xy[-2]

    norm = np.hypot(dx, dy)
    nx, ny = -dy / norm, dx / norm          # unit left normals

    outer = xy + half[:, None] * np.column_stack((nx, ny))
    inner = xy - half[:, None] * np.column_stack((nx, ny))

    # 4 ── stitch polygon ---------------------------------------------------------
    if closed:
        poly = np.vstack([outer, outer[0], inner[0], inner[::-1]])
    else:
        # rectangular caps using end‑segment tangents
        cap_end   = xy[-1] + half[-1] * np.array([-dx[-1], -dy[-1]]) / norm[-1]
        cap_start = xy[0]  + half[0]  * np.array([dx[0],  dy[0]])  / norm[0]
        poly = np.vstack([outer,
                          cap_end[None, :],
                          inner[::-1],
                          cap_start[None, :]])

    # 5 ── build component --------------------------------------------------------
    c = gf.Component()
    c.add_polygon(poly, layer=layer)

    return c, poly

def correct_euler(radius, angle, p, resolution, use_eff=True, N=1000):
    turn = gf.path.euler(radius=radius, angle=angle, p=p, use_eff=use_eff, npoints=N)
    
    n2 = int(turn.length()/resolution * angle/360)
    path_euler = gf.path.euler(radius=radius, angle=angle, p=p, use_eff=use_eff, npoints=n2)
    return path_euler

class bend5:
    def __init__(self, θ1, k1, k1p, x1, y1):
        self.θ1 = θ1
        self.k1 = k1
        self.k1p = k1p
        self.x1 = x1
        self.y1 = y1

    def compute_integral(self, s, theta, a1, a2, a3):

        def integrand(x):
            phase = theta + 0.5 * a1 * x**2 + (1/3) * a2 * x**3 + 0.25 * a3 * x**4
            return np.exp(1j * phase)
        
        # Separate the real and imaginary parts
        real_integrand = lambda x: np.real(integrand(x))
        imag_integrand = lambda x: np.imag(integrand(x))
        
        # Integrate real and imaginary parts separately
        real_result, real_error = quad(real_integrand, 0, s, limit=500, epsabs=1e-8, epsrel=1e-8)
        imag_result, imag_error = quad(imag_integrand, 0, s, limit=500, epsabs=1e-8, epsrel=1e-8)

        # Combine the results into a complex number
        return real_result, imag_result

    def system_equations(self, vars):
        θ0, s1, a1, a2, a3 = vars
        eqn1 = θ0 + a1*s1**2/2 + a2*s1**3/3 + a3*s1**4/4 - self.θ1
        eqn2 = a1*s1 + a2*s1**2 + a3*s1**3 - self.k1
        eqn3 = a1 + 2*a2*s1 + 3*a3*s1**2 - self.k1p
        x1c, y1c = self.compute_integral(s1, θ0, a1, a2, a3)
        eqn4 = x1c - self.x1
        eqn5 = y1c - self.y1
        return [eqn1, eqn2, eqn3, eqn4, eqn5]
    
    def evaluate(self, initial_guess, output=True):
        solution = fsolve(self.system_equations, initial_guess, xtol=1e-8)
        self.θ0, self.s1, self.a1, self.a2, self.a3 = solution
        residuals = self.system_equations(solution)
        if output:
            print("Residuals:", residuals)
            print("solution:", [self.θ0, self.s1, self.a1, self.a2, self.a3])
        return solution

    def draw_path(self, N=1000, plot=True):
        s_array = np.linspace(0, self.s1, N)
        x_array, y_array = [], []
        for s in s_array:
            x, y = self.compute_integral(s, self.θ0, self.a1, self.a2, self.a3)
            x_array.append(x)
            y_array.append(y)
        if plot:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(np.array(x_array), np.array(y_array))
            plt.show()
        else:
            return x_array, y_array

    def draw_curvature(self, N=1000):
        s_array = np.linspace(0, self.s1, N)
        k_array = self.a1*s_array + self.a2*s_array**2 + self.a3*s_array**3
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(s_array, k_array)
        plt.show()


class bend8:
    def __init__(self, θ0, k0, k0p, θ1, k1, k1p, x1, y1):
        self.θ0 = θ0
        self.k0 = k0
        self.k0p = k0p
        self.θ1 = θ1
        self.k1 = k1
        self.k1p = k1p
        self.x1 = x1
        self.y1 = y1

    def compute_integral(self, s, a2, a3, a4, a5):

        def integrand(x):
            phase = self.θ0 + self.k0*x + self.k0p*x**2/2 + a2*x**3/3 + a3*x**4/4 + a4*x**5/5 + a5*x**6/6
            return np.exp(1j * phase)
        
        # Separate the real and imaginary parts
        real_integrand = lambda x: np.real(integrand(x))
        imag_integrand = lambda x: np.imag(integrand(x))
        
        # Integrate real and imaginary parts separately
        real_result, real_error = quad(real_integrand, 0, s, limit=10000, epsabs=1e-8, epsrel=1e-8)
        imag_result, imag_error = quad(imag_integrand, 0, s, limit=10000, epsabs=1e-8, epsrel=1e-8)
        
        # Combine the results into a complex number
        return real_result, imag_result

    def system_equations(self, vars):
        s1, a2, a3, a4, a5 = vars
        eqn1 = self.θ0 + self.k0*s1 + self.k0p*s1**2/2 + a2*s1**3/3 + a3*s1**4/4 + a4*s1**5/5 + a5*s1**6/6 - self.θ1
        eqn2 = self.k0 + self.k0p*s1 + a2*s1**2 + a3*s1**3 + a4*s1**4 + a5*s1**5 - self.k1
        eqn3 = self.k0p + 2*a2*s1 + 3*a3*s1**2 + 4*a4*s1**3 + 5*a5*s1**4 - self.k1p
        x1c, y1c = self.compute_integral(s1, a2, a3, a4, a5)
        eqn4 = x1c - self.x1
        eqn5 = y1c + self.y1

        return [eqn1, eqn2, eqn3, eqn4, eqn5]
    
    def evaluate(self, initial_guess, output=True):
        solution = fsolve(self.system_equations, initial_guess, xtol=1e-8)
        self.s1, self.a2, self.a3, self.a4, self.a5 = solution
        residuals = self.system_equations(solution)
        if output:
            print("Residuals:", residuals)
            print("solution:", [self.s1, self.a2, self.a3, self.a4, self.a5])
        return solution

    def draw_path(self, N=1000, plot=True):
        s_array = np.linspace(0, self.s1, N)
        x_array, y_array = [], []
        for s in s_array:
            x, y = self.compute_integral(s, self.a2, self.a3, self.a4, self.a5)
            x_array.append(x)
            y_array.append(y)
        if plot:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(np.array(x_array), np.array(y_array))
            plt.show()
        else:
            return np.array(x_array), np.array(y_array)

    def draw_curvature(self, N=1000):
        s_array = np.linspace(0, self.s1, N)
        k_array = self.k0 + self.k0p*s_array + self.a2*s_array**2 + self.a3*s_array**3 + self.a4*s_array**4 + self.a5*s_array**5
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(s_array, k_array)
        plt.show()


def bend5_initial_guess_iter(l, h, θ1=0, k1=0, k1p=0,steps=10, output=False):
    h0 = 20
    l0=  100
    θ10 = 0
    k10 = 0
    k10p = 0
    initial_guess = [0.49436916287540195, 103.56472149846847, -0.0005531067206873361, 1.0681373204744205e-05, -5.1568589420900665e-08]
    # initial_guess = [np.float64(1.7293446046920555), np.float64(779.4020680837679), np.float64(-3.416171629897188e-05), np.float64(8.766134373482581e-08), np.float64(-5.623627863231646e-11)]
    h_arr = np.linspace(h0, h, steps)
    l_arr = np.linspace(l0, l, steps)
    θ1_arr = np.linspace(θ10, θ1, steps)
    k1_arr = np.linspace(k10, k1, steps)
    k1p_arr = np.linspace(k10p, k1p, steps)

    for s in range(steps):
        solver = bend5(θ1_arr[s], k1_arr[s], k1p_arr[s], l_arr[s], h_arr[s])
        solution = solver.evaluate(initial_guess, output=False)
        initial_guess = [solver.θ0, solver.s1, solver.a1, solver.a2, solver.a3]

    if output:
        print(initial_guess)
    return solver

def bend8_initial_guess_iter(l, h, θ0=0, k0=0, k0p=0, θ1=0, k1=0, k1p=0, steps=10, output=False):
    h0 = 10
    l0=  280
    θ00 = 0
    k00 = 0
    k0p0 = 0
    θ10 = -np.pi
    k10 = 0
    k1p0 = 0

    initial_guess = [403.9330690840626, 2.1700104743235264e-06, -3.11213851862657e-08, 1.1370101463262695e-10, -1.2385911631149373e-13]

    h_arr = np.linspace(h0, h, steps)
    l_arr = np.linspace(l0, l, steps)
    θ0_arr = np.linspace(θ00, θ0, steps)
    k0_arr = np.linspace(k00, k0, steps)
    k0p_arr = np.linspace(k0p0, k0p, steps)
    θ1_arr = np.linspace(θ10, θ1, steps)
    k1_arr = np.linspace(k10, k1, steps)
    k1p_arr = np.linspace(k1p0, k1p, steps)

    for s in range(steps):
        solver = bend8(θ0_arr[s], k0_arr[s], k0p_arr[s], θ1_arr[s], k1_arr[s], k1p_arr[s], l_arr[s], -h_arr[s])
        solution = solver.evaluate(initial_guess, output=False)
        initial_guess = [solver.s1, solver.a2, solver.a3, solver.a4, solver.a5]

    if output:
        print(initial_guess)
    return solver

def pulley_coupler_connector(l1=100, l2=100, d1=20, d2=20, θ1=np.pi/6, θ2=np.pi/6, Rc=50, Rc0=None, steps=20, resolution=1, output=True):

    if Rc0 is None:
        solver_in = bend8_initial_guess_iter(l1, Rc*np.cos(θ1)-d1, 0, 0, 0, θ1, -1/Rc, 0, steps=steps, output=output)
    else:
        solver_in = bend8_initial_guess_iter(l1, Rc*np.cos(θ1)-d1, 0, 0, 0, θ1, -1/Rc0, 0, steps=steps, output=output)

    x_pos_in, y_pos_in = solver_in.draw_path(plot=False)
    dx = np.diff(x_pos_in)
    dy = np.diff(y_pos_in)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)
    length_s = np.sum(ds)   
    n = int(length_s/resolution)
    x_pos_in, y_pos_in = solver_in.draw_path(n, plot=False)
    x_drift_in, y_drift_in = -Rc*np.sin(θ1) - l1, d1

    if Rc0 is None:
        solver_out = bend8_initial_guess_iter(l2, d2-Rc*np.cos(θ2), -θ2, -1/Rc, 0, 0, 0, 0, steps=steps, output=output)
    else:
        solver_out = bend8_initial_guess_iter(l2, d2-Rc*np.cos(θ2), -θ2, -1/Rc0, 0, 0, 0, 0, steps=steps, output=output)
    
    x_pos_out, y_pos_out = solver_out.draw_path(plot=False)
    dx = np.diff(x_pos_out)
    dy = np.diff(y_pos_out)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)
    length_s = np.sum(ds)
    n = int(length_s/resolution)
    x_pos_out, y_pos_out = solver_out.draw_path(n, plot=False)
    x_drift_out, y_drift_out = Rc*np.sin(θ2), Rc*np.cos(θ2)

    n = int(Rc*(θ1+θ2)/resolution)
    if n % 2 == 0:  # Ensure n is odd for better symmetry
        n += 1
    θ_array = np.linspace(-θ1, θ2, n)

    x_pos_c = np.sin(θ_array) * Rc
    y_pos_c = np.cos(θ_array) * Rc
    
    x_combined = np.concatenate((x_pos_in+x_drift_in, x_pos_c[1:], x_pos_out[1:]+x_drift_out))
    y_combined = np.concatenate((y_pos_in+y_drift_in, y_pos_c[1:], y_pos_out[1:]+y_drift_out))
    return x_combined, y_combined

def spiral_inner_connector(length = 400, gap = 100, steps=10, resolution=1, output=True):
    solver = bend5_initial_guess_iter(length/2, gap/2, steps=steps, output=output)
    x_pos, y_pos = solver.draw_path(plot=False)
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    dx = np.diff(x_pos)
    dy = np.diff(y_pos)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)

    length = np.sum(ds)
    n = int(length/resolution)

    x_pos, y_pos = solver.draw_path(n, plot=False)
    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)

    x_combined = np.concatenate((-x_pos[::-1][:-1], x_pos))
    y_combined = np.concatenate((-y_pos[::-1][:-1], y_pos))
    return x_combined, y_combined

def spiral_outer_connector(left_length = 50, right_length = 10, gap = 100, steps=10, resolution = 1, output=True):
    solver = bend8_initial_guess_iter(l=left_length, h=-gap, θ1=-np.pi, steps=steps, output=output)
    x_pos, y_pos = solver.draw_path(plot=False)

    dx = np.diff(x_pos)
    dy = np.diff(y_pos)
    ds = np.sqrt((dx) ** 2 + (dy) ** 2)
    length = np.sum(ds)
    n = int(length/resolution)
    x_pos, y_pos = solver.draw_path(n, plot=False)

    n_straight = int((right_length+x_pos[-1])/resolution)
    x_straight = np.linspace(right_length, -x_pos[-1], n_straight)
    y_straight = np.linspace(gap, -y_pos[-1], n_straight)

    n_right_straight = int(right_length/resolution)
    x_right_straight = np.linspace(-x_pos[0], right_length, n_right_straight)
    y_right_straight = np.linspace(-y_pos[0], 0, n_right_straight)

    if n_right_straight == 0 and right_length > 0:
        x_right_straight = np.array([-x_pos[0], right_length])
        y_right_straight = np.array([-y_pos[0], 0])

    x_combined = np.concatenate((x_straight[:-1], -x_pos[::-1], x_right_straight[1:]))
    y_combined = np.concatenate((y_straight[:-1], -y_pos[::-1], y_right_straight[1:]))

    return x_combined, y_combined

