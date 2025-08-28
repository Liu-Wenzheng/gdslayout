import os
import yaml
import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree
from scipy.signal import savgol_filter
import networkx as nx

import gdsfactory as gf
from gdsfactory.port import Port

from . import init_component
from .device import device_registry, coupler_registry, ensure_discovered

def _get_derivatives(polyline, idx, is_closed=False):
    """Helper to get first and second derivatives at an index using finite differences."""
    n = len(polyline)
    
    if is_closed:
        n_unique = n - 1
        p_prev = polyline[(idx - 1 + n_unique) % n_unique]
        p_curr = polyline[idx % n_unique]
        p_next = polyline[(idx + 1) % n_unique]
    else:
        if idx == 0:
            p_prev, p_curr, p_next = polyline[0], polyline[1], polyline[2]
        elif idx == n - 1:
            p_prev, p_curr, p_next = polyline[n-3], polyline[n-2], polyline[n-1]
        else:
            p_prev, p_curr, p_next = polyline[idx-1], polyline[idx], polyline[idx+1]

    first_derivative = (p_next - p_prev) / 2.0
    second_derivative = p_next - 2 * p_curr + p_prev
    
    return first_derivative, second_derivative


def _get_curvature_radius(polyline, idx, is_closed=False):
    """Calculates the radius of curvature at a specific index of a polyline."""
    v, a = _get_derivatives(polyline, idx, is_closed)
    
    numerator = np.abs(v[0] * a[1] - v[1] * a[0])
    denominator = np.linalg.norm(v)**3
    
    if denominator < 1e-9:
        return np.inf
        
    curvature = numerator / denominator
    
    if curvature < 1e-9:
        return np.inf
        
    return 1.0 / curvature


def _get_tangent_angle(polyline, idx, is_closed=False):
    """Calculates the tangent vector angle (in degrees) at a specific index of a polyline."""
    first_derivative, _ = _get_derivatives(polyline, idx, is_closed)
    angle_rad = np.arctan2(first_derivative[1], first_derivative[0])
    return np.degrees(angle_rad)


def compute_midline(points1, points2, initial_idx=(0, 0), terminator_width=np.inf,
                    smoothing_window=31, smoothing_order=3):
    """
    Computes a smoothed midline between two polylines using a Voronoi diagram
    and returns endpoint analysis.
    """
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError("Input points must be 2-dimensional for this function.")

    # Auto-detect if points1 is a closed curve
    is_closed1 = np.allclose(points1[0], points1[-1])

    # Voronoi diagram and graph construction
    pts = np.vstack([points1, points2])
    labels = np.hstack([np.zeros(len(points1), dtype=bool), np.ones(len(points2), dtype=bool)])
    vor = Voronoi(pts)
    G = nx.Graph()
    for (p, q), v_indices in zip(vor.ridge_points, vor.ridge_vertices):
        if labels[p] == labels[q] or -1 in v_indices: continue
        v1, v2 = vor.vertices[v_indices[0]], vor.vertices[v_indices[1]]
        G.add_edge(tuple(v1), tuple(v2), weight=np.linalg.norm(v1 - v2))

    if G.number_of_edges() == 0: raise ValueError("No separating Voronoi ridges found.")

    # Find the principal path through the graph
    components = list(nx.connected_components(G))
    if not components: raise ValueError("Midline graph is empty.")
    largest_component_nodes = max(components, key=len)
    G_sub = G.subgraph(largest_component_nodes)
    p0 = (points1[initial_idx[0]] + points2[initial_idx[1]]) / 2.0
    nodes_array = np.array(list(G_sub.nodes()))
    start_node = tuple(nodes_array[np.argmin(np.linalg.norm(nodes_array - p0, axis=1))])
    path_lengths_from_start = nx.shortest_path_length(G_sub, source=start_node, weight='weight')
    endpoint1 = max(path_lengths_from_start, key=path_lengths_from_start.get)
    path_lengths_from_endpoint1 = nx.shortest_path_length(G_sub, source=endpoint1, weight='weight')
    endpoint2 = max(path_lengths_from_endpoint1, key=path_lengths_from_endpoint1.get)
    longest_path_nodes = nx.shortest_path(G_sub, source=endpoint1, target=endpoint2, weight='weight')
    mid_pts_full = np.array(longest_path_nodes)

    # Trim the path using the terminator width
    kd1, kd2 = cKDTree(points1), cKDTree(points2)
    dists_full, _ = kd1.query(mid_pts_full)
    start_idx_final = np.argmin(np.linalg.norm(mid_pts_full - p0, axis=1))
    
    final_mid_pts, final_dists = [mid_pts_full[start_idx_final]], [dists_full[start_idx_final]]
    for i in range(start_idx_final + 1, len(mid_pts_full)):
        if dists_full[i] > terminator_width: break
        final_mid_pts.append(mid_pts_full[i]); final_dists.append(dists_full[i])
    for i in range(start_idx_final - 1, -1, -1):
        if dists_full[i] > terminator_width: break
        final_mid_pts.insert(0, mid_pts_full[i]); final_dists.insert(0, dists_full[i])
    mid_pts_raw, dists_raw = np.array(final_mid_pts), np.array(final_dists)

    # Smooth the final midline points and distances
    if len(mid_pts_raw) > smoothing_window:
        mid_x_smooth = savgol_filter(mid_pts_raw[:, 0], smoothing_window, smoothing_order)
        mid_y_smooth = savgol_filter(mid_pts_raw[:, 1], smoothing_window, smoothing_order)
        mid_pts_smooth = np.vstack((mid_x_smooth, mid_y_smooth)).T
        dists_smooth = savgol_filter(dists_raw, smoothing_window, smoothing_order)
    else:
        mid_pts_smooth, dists_smooth = mid_pts_raw, dists_raw

    # Analyze the endpoints
    endpoint_info = {}
    if len(mid_pts_smooth) > 1:
        left_end_pt, right_end_pt = mid_pts_smooth[0], mid_pts_smooth[-1]
        
        _, idx1_left = kd1.query(left_end_pt)
        _, idx2_left = kd2.query(left_end_pt)
        angle1_left = _get_tangent_angle(points1, idx1_left, is_closed=is_closed1)
        angle2_left = _get_tangent_angle(points2, idx2_left)
        radius1_left = _get_curvature_radius(points1, idx1_left, is_closed=is_closed1)
        radius2_left = _get_curvature_radius(points2, idx2_left)

        _, idx1_right = kd1.query(right_end_pt)
        _, idx2_right = kd2.query(right_end_pt)
        angle1_right = _get_tangent_angle(points1, idx1_right, is_closed=is_closed1)
        angle2_right = _get_tangent_angle(points2, idx2_right)
        radius1_right = _get_curvature_radius(points1, idx1_right, is_closed=is_closed1)
        radius2_right = _get_curvature_radius(points2, idx2_right)

        endpoint_info = {
            'left_pts': 1e-6 * np.array([points1[idx1_left], points2[idx2_left]]), 'right_pts': 1e-6 * np.array([points1[idx1_right], points2[idx2_right]]),
            'left_angles_deg': 180 + np.array([angle1_left, angle2_left]), 'right_angles_deg': np.array([angle1_right, angle2_right]),
            'left_radii': 1e-6 * np.array([radius1_left, radius2_left]), 'right_radii': 1e-6 * np.array([radius1_right, radius2_right])
        }

    # Final verification
    dists2_raw, _ = kd2.query(mid_pts_raw)
    assert np.allclose(dists_raw, dists2_raw, atol=1e-5), "Geometric distance mismatch detected before smoothing."

    return mid_pts_smooth, dists_smooth, endpoint_info


def simport_info(points1, points2, initial_idx=(0, 0), terminator_width=40.0):
    _, _, endpoint_info = compute_midline(points1, points2, initial_idx, terminator_width)
    return endpoint_info


class Structure:
    
    def __init__(self, config_path=None, config_id=None, config_folder=None):
        """
        Initialize the Structure class.
        
        Args:
            config_path (str, optional): Path to YAML configuration file
            config_id (int, optional): Configuration ID
            config_folder (str, optional): Path to the folder containing configuration files

        Note: Either config_path or config_id must be provided.
        """

        # gf.clear_cache()
        self.config_path = config_path
        self.config_id = config_id
        self.config_folder = config_folder
        
        self.load_from_file()
        self.device = None
        self.device_path = None
        self.coupler = None
        self.coupler_path = None
        self.component = None
        self.port_coupling = 'gap' not in self.config

        if isinstance(self.config['device'], list):
            self.num_devices = len(self.config['device'])
            self.id = self.config['coupler_attach_device_id']
        elif isinstance(self.config['device'], dict):
            self.num_devices = 1
        else:
            raise ValueError("Invalid device configuration format.")

    def load_from_file(self):

        if self.config_path is not None:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        elif self.config_id is not None and self.config_folder is not None:
            with open(os.path.join(self.config_folder, f"{self.config_id}.yaml"), 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            raise ValueError("Missing 'device' or 'devices' section in configuration")
        
    def _load_device(self, details=False):
        """
        Load the device component(s) based on the configuration.
        Auto-detects single device vs multiple devices format.
        
        Returns:
            tuple: (device_component, device_path), device_offset
        """

        ensure_discovered()

        if details:
            print(f"Auto-detected {self.num_devices} devices in configuration")

        def _inject_layer(cfg: dict) -> dict:
            cfg = copy.deepcopy(cfg)
            cfg['layer'] = self.config['layer']
            return cfg
        
        if details:
            print(f"Auto-detected {self.num_devices} devices in configuration")

        if self.num_devices == 1:

            device_config = _inject_layer(self.config['device'])
            device_type = device_config.pop('type')

            if details:
                print(f"Loading single device type: {device_type}")
                print(f"Device config: {device_config}")
            try:
                provider = device_registry.resolve(device_type)
            except KeyError as e:
                raise ValueError(f"Unknown device type: {device_type}. "
                                 f"Available: {device_registry.names()}") from e
            return provider(**device_config)
        else:
            self.devices = []
            combined_paths = []
            
            for i, device_data in enumerate(self.config['device']):
                device_config = _inject_layer(device_data)
                device_type = device_config.pop('type')
                
                if details:
                    print(f"Loading device {i+1}/{self.num_devices} - type: {device_type}")
                    print(f"Device config: {device_config}")

                try:
                    provider = device_registry.resolve(device_type)
                except KeyError as e:
                    raise ValueError(f"Unknown device type: {device_type}. "
                                     f"Available: {device_registry.names()}") from e

                device_component, device_path = provider(**device_config)
                self.devices.append(device_component)
                combined_paths.append(device_path)

            device = gf.Component()
            device_refs = []
            for idx, dev in enumerate(self.devices):
                device_ref = device.add_ref(dev)
                device_refs.append(device_ref)
                if idx == self.config.get('coupler_attach_device_id', 0):
                    device.add_ports(device_ref.ports)
            
            self._connect_devices(device_refs)
            return (device, combined_paths)
        
    
    def _get_devices_id(self):
        if self.num_devices == 1:
            return self.device
        else:
            return self.devices[self.id]

    
    def _load_coupler(self, details=False):
        """
        Load the coupler component based on the configuration.
        
        Returns:
            tuple: (coupler_component, coupler_path), gap_distance
        """
    
        ensure_discovered()

        coupler_config = copy.deepcopy(self.config['coupler'])

        if not self.port_coupling:
            coupler_config_list = [coupler_config]
        else:
            coupler_config_list = coupler_config
            coupler_component_list = []
            coupler_path_list = []

        for coupler_config in coupler_config_list:
            coupler_config['layer'] = self.config['layer']

            coupler_type = coupler_config.pop('type')
            try:
                coupler_provider = coupler_registry.resolve(coupler_type)
            except KeyError as e:
                raise ValueError(f"Unknown coupler type: {coupler_type}. "
                                 f"Available: {coupler_registry.names()}") from e

            if self.device is not None:
                coupler_config['center'] = (
                    self._get_devices_id().ports['coupler'].x,
                    self._get_devices_id().ports['coupler'].y + self.config.get('gap', 0) + coupler_config.get('width', 1)/2
                ) if not self.port_coupling else self._get_devices_id().ports['coupler'].center
            
            # Handle ring_down configuration
            if coupler_config.get('ring_down') is not None:
                l, d = coupler_config['ring_down']
                if isinstance(l, str) and self.device is not None:
                    if l.startswith('r:'):
                        ratio = float(l.split(':')[1].strip())
                        l = self.device.xsize * ratio
                    else:
                        raise ValueError("Invalid format for 'ring_down' length. Expected 'r:<ratio>' or a numeric value.")
                elif not isinstance(l, (float, int)):
                    l = float(l)
                
                if isinstance(d, str) and self.device is not None:

                    if d == 'm':  # Middle
                        d = - self.device.y
                    elif d == 'b':  # Top
                        d = - self.device.ymin
                    elif d == 't':  # Bottom
                        d = - self.device.ymax
                    elif d.startswith('b:'):  # Top + offset
                        offset = float(d.split(':')[1].strip())
                        d = - self.device.ymin - offset
                    elif d.startswith('t:'):  # Bottom + offset
                        offset = float(d.split(':')[1].strip())
                        d = - self.device.ymax - offset
                    elif d.startswith('m:'):  # Middle + offset
                        offset = float(d.split(':')[1].strip())
                        d = - self.device.y - offset
                    elif d.startswith('r:'):  # Relative to device height
                        ratio = float(d.split(':')[1].strip())
                        d = - self.device.ymax + self.device.ysize * ratio
                    else:
                        raise ValueError(f"Invalid format for 'ring_down' depth: {d}. Expected 'm', 't', 'b', 't:<offset>', 'b:<offset>', 'm:<offset>', or 'r:<ratio>'.")
                elif not isinstance(d, (float, int)):
                    d = float(d)

                coupler_config['ring_down'] = (l, d)

            # Handle etch_info configuration
            if coupler_config.get('etch_info') is not None and self.device is not None:
                upper, lower = coupler_config['etch_info']
                if upper == 't':
                    upper = self._get_devices_id().ymax - self._get_devices_id().ports['coupler'].y
                elif isinstance(upper, (float, int)):
                    upper += coupler_config.get('width', 1.0) / 2
                else:
                    raise ValueError(f"Invalid format for 'etch_info' upper value: {upper}. Expected a float or 't'.")
                if lower == 'b':
                    lower = self._get_devices_id().ports['coupler'].y - self._get_devices_id().ymin
                elif isinstance(lower, (float, int)):
                    lower += coupler_config.get('width', 1.0) / 2
                else:
                    raise ValueError(f"Invalid format for 'etch_info' lower value: {lower}. Expected a float or 'b'.")
                coupler_config['etch_info'] = (upper, lower)

            if details:
                print(f"Loading coupler type: {coupler_type}")
                print(f"Coupler config: {coupler_config}")

            if not self.port_coupling:
                return coupler_provider(**coupler_config)
            else:
                coupler_component, coupler_path = coupler_provider(**coupler_config)
                coupler_component_list.append(coupler_component)
                coupler_path_list.append(coupler_path)

        return (coupler_component_list, coupler_path_list)

    def _connect_devices(self, device_refs):
        """Connect devices based on device_connections configuration."""
        if 'device_connections' not in self.config:
            return
        
        for connection in self.config['device_connections']:
            from_device_id, from_ports = self._parse_connection_spec(connection.get('from', ''))
            to_device_id, to_ports = self._parse_connection_spec(connection.get('to', ''))
        
            self._connect_device_ports(device_refs[from_device_id], from_ports, 
                                        device_refs[to_device_id], to_ports)
    
    def _parse_connection_spec(self, spec):
        """Parse 'device[N].ports[port_spec]' format."""
        import re
        match = re.match(r'device\[(\d+)\]\.ports\[[\'"]?([^\'"]*)[\'"]?\]', spec)
        if not match:
            raise ValueError(f"Invalid connection specification: {spec}")
        device_id = int(match.group(1))
        port_names = match.group(2)

        return device_id, port_names
    
    def _connect_device_ports(self, from_device_ref, from_ports, to_device_ref, to_ports):
        """Connect ports between two device references."""
        try:
            # Get available port names
            from_available = [port.name for port in from_device_ref.ports] if hasattr(from_device_ref.ports, '__iter__') else list(from_device_ref.ports.keys())
            to_available = [port.name for port in to_device_ref.ports] if hasattr(to_device_ref.ports, '__iter__') else list(to_device_ref.ports.keys())

            if (from_ports in from_available) and (to_ports in to_available):
                try:
                    from_device_ref.connect(from_ports, to_device_ref.ports[to_ports])
                    print(f"✅ Connected {from_ports} -> {to_ports}")
                except Exception as e:
                    print(f"⚠️  Connection failed {from_ports} -> {to_ports}: {e}")

        except Exception as e:
            print(f"Warning: Could not connect ports: {e}")

    def build(self, cell=None, details=False, force_rebuild=False):
        """
        Build the complete structure by loading and combining device and coupler.
        
        Returns:
            gf.Component: The complete structure component
        """
        if self.component is not None and not force_rebuild:
            return self.component

        self.device, self.device_path = self._load_device(details=details)
        self.coupler, self.coupler_path = self._load_coupler(details=details)

        if cell == "toplevel":
            if gf.kcl.has_cell(cell):
                cell_to_delete = gf.kcl[cell]
                gf.kcl.delete_cell(cell_to_delete)
            component = gf.Component(cell)
        else:
            component = gf.Component()

        port_mapping = {}

        device_ref = component.add_ref(self.device)
        port_mapping['coupler'] = device_ref.ports['coupler']

        if not self.port_coupling:
            coupler_ref = component.add_ref(self.coupler)
            port_mapping['in'] = coupler_ref.ports['in']
            port_mapping['out'] = coupler_ref.ports['out']

        else:
            coupler_ref0 = component.add_ref(self.coupler[0])
            coupler_ref0.connect('out', device_ref.ports['in']) if self.num_devices == 1 else coupler_ref0.connect('out', self.devices[self.id].ports['in'])
            self.coupler_path[0].move([-self.coupler[0].ports['out'].center[i] + device_ref.ports['in'].center[i] for i in range(2)])
            coupler_ref1 = component.add_ref(self.coupler[1])
            coupler_ref1.connect('in', device_ref.ports['out']) if self.num_devices == 1 else coupler_ref1.connect('in', self.devices[self.id].ports['out'])
            self.coupler_path[1].move([-self.coupler[1].ports['in'].center[i] + device_ref.ports['out'].center[i] for i in range(2)])
            port_mapping['in'] = coupler_ref0.ports['in']
            port_mapping['out'] = coupler_ref1.ports['out']

        component.ports.clear()
        for name, port in port_mapping.items():
            component.add_port(name=name, port=port)
        component.add_port(name='device_center', center=device_ref.center, width=1, orientation=0, layer=self.config['layer'])
        self.component = component
        return self.component
    

    def lumerical(self, decoupling_width=4.0, auto_compute='wg_coupling', plot=False):
        """
        Generate Lumerical script for the structure.
        
        Args:
            plot (bool): If True, generate a plot of the structure.
            
        Returns:
            str: Lumerical script
        """
        component = self.build()

        if hasattr(self.device_path, 'points'):
            points1 = self.device_path.points
        elif isinstance(self.device_path, list):
            points1 = np.concatenate([path.points if hasattr(path, 'points') else np.array(path) for path in self.device_path])
        else:
            ValueError("Invalid device path format")

        if hasattr(self.coupler_path, 'points'):
            points2 = self.coupler_path.points
        elif isinstance(self.coupler_path, list):
            points2 = np.concatenate([path.points if hasattr(path, 'points') else np.array(path) for path in self.coupler_path])
        else:
            ValueError("Invalid coupler path format")

        if plot:
    
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.scatter(points1[:, 0], points1[:, 1], color='blue', s=1, alpha=0.3)
            ax.scatter(points2[:, 0], points2[:, 1], color='orange', s=1, alpha=0.3)
            
            if auto_compute == 'wg_coupling':
                idx2 = np.where(np.isclose(points2[:, 0], self.device.ports['coupler'].x))[0]
                ax.scatter(points1[0, 0], points1[0, 1], color='blue', s=10, label='Device Points')
                ax.scatter(points2[idx2, 0], points2[idx2, 1], color='orange', s=10, label='Coupler Points')
                center, w, info = compute_midline(points1, points2, initial_idx=(0, idx2), terminator_width=4.0)
                ax.plot(center[:, 0], center[:, 1], label='Midline', color='green', linewidth=2)
                if info:
                    p1_left, p2_left = info['left_pts']
                    p1_right, p2_right = info['right_pts']
                    
                    endpoint_correspondents = np.array([p1_left, p2_left, p1_right, p2_right])

                    ax.scatter(endpoint_correspondents[:, 0], endpoint_correspondents[:, 1],
                                color='red', 
                                s=10,             
                                edgecolors='black', 
                                zorder=5,          
                                label='Endpoint Correspondents')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_title('Device and Coupler Paths with Midline Endpoints')
            ax.grid(True)
            ax.set_aspect('equal')
            return ax
        else:
            if auto_compute == 'wg_coupling':
                idx2 = np.where(np.isclose(points2[:, 0], self.device.ports['coupler'].x))[0]
                info = simport_info(points1, points2, initial_idx=(0, idx2), terminator_width=decoupling_width)
            elif auto_compute == 'phc':
                info = {
                    
                }
            else:
                raise ValueError(f"Unknown auto_compute option: {auto_compute}. Expected 'wg_coupling' or 'phc'.")

        info['xmin'] = component.xmin * 1e-6
        info['xmax'] = component.xmax * 1e-6
        info['ymin'] = component.ymin * 1e-6
        info['ymax'] = component.ymax * 1e-6
        info['xsize'] = component.xsize * 1e-6
        info['ysize'] = component.ysize * 1e-6

        return info, component

    def create_gds(self, gds_filename='../structure.gds', lumerical_plugins=False, auto_compute='wg_coupling', decoupling_width=4.0):
        if lumerical_plugins:
            info, component = self.lumerical(decoupling_width=decoupling_width, auto_compute=auto_compute, plot=False)
            component.write_gds(gdspath=gds_filename)
            return info
        else:
            if 'coupler' not in self.config:
                (device, path) = self._load_device()
                device.write_gds(gdspath=gds_filename)
                return gds_filename
            if 'device' not in self.config:
                (coupler, path) = self._load_coupler()
                coupler.write_gds(gdspath=gds_filename)
                return gds_filename
            if 'device' in self.config and 'coupler' in self.config:
                component = self.build()
                component.write_gds(gdspath=gds_filename)
                return gds_filename