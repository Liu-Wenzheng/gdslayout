import gdsfactory as gf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gdsfactory.component import Component
import numpy as np
from collections import defaultdict

def plot_component(component: Component, with_ports: bool = True, with_labels: bool = True, 
                  figsize: tuple = (12, 10)):
    """
    Enhanced component visualization with modern styling.
    
    Args:
        component: The gdsfactory component to plot
        with_ports: Show port markers and labels
        with_labels: Show component labels
        figsize: Figure size (width, height)
    """
    # Setup plot with modern styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafafa')
    
    # Constants
    SCALE_FACTOR = 1000.0  # nm to μm conversion
    coords = {'x': [], 'y': []}  # Coordinate collector
    
    # Modern color scheme with high contrast colors
    colors = ['#1B4F72', '#145A32', '#B7950B', '#A93226', '#6C3483', '#117A65']
    
    # 1. Extract and plot geometry efficiently
    geometry_plotted = _plot_geometry(component, ax, colors, SCALE_FACTOR, coords)
    
    # 2. Plot labels if requested
    if with_labels:
        _plot_labels(component, ax, SCALE_FACTOR, coords)
    
    # 3. Plot ports with enhanced visualization
    if with_ports:
        _plot_ports(component, ax, coords)
    
    # 4. Finalize plot with modern styling
    _finalize_plot(ax, coords, component.name, geometry_plotted, with_ports, with_labels)
    
    return fig, ax

def _plot_geometry(component, ax, colors, scale_factor, coords):
    """Extract and plot component geometry efficiently."""
    try:
        polygons = component.get_polygons()
        if not polygons:
            return False
            
        for layer_idx, (layer, polys) in enumerate(polygons.items()):
            color = colors[layer_idx % len(colors)]
            
            for poly in polys:
                points = _extract_points(poly, scale_factor)
                if len(points) >= 3:
                    # Efficient coordinate collection
                    x_coords, y_coords = zip(*points)
                    coords['x'].extend(x_coords)
                    coords['y'].extend(y_coords)
                    
                    # Create polygon with modern styling
                    patch = patches.Polygon(points, closed=True, facecolor=color, 
                                          edgecolor='black', alpha=0.9, 
                                          linewidth=1.5, zorder=1)
                    ax.add_patch(patch)
                elif len(points) == 2:
                    # Handle line segments
                    x_vals, y_vals = zip(*points)
                    ax.plot(x_vals, y_vals, color=color, linewidth=4, alpha=0.9, zorder=1)
                    coords['x'].extend(x_vals)
                    coords['y'].extend(y_vals)
        return True
    except Exception as e:
        print(f"Geometry extraction failed: {e}")
        return False

def _extract_points(poly, scale_factor):
    """Extract points from various polygon formats efficiently."""
    if hasattr(poly, 'each_point_hull'):
        # KLayout polygon
        return [(float(pt.x) / scale_factor, float(pt.y) / scale_factor) 
                for pt in poly.each_point_hull()]
    elif hasattr(poly, 'exterior'):
        # Shapely polygon
        return [(p[0] / scale_factor, p[1] / scale_factor) 
                for p in poly.exterior.coords]
    elif hasattr(poly, '__len__') and len(poly) >= 2:
        # Array of points
        if hasattr(poly[0], '__len__'):
            return [(p[0] / scale_factor, p[1] / scale_factor) for p in poly]
        elif len(poly) % 2 == 0:
            return [(poly[i] / scale_factor, poly[i+1] / scale_factor) 
                    for i in range(0, len(poly), 2)]
    return []

def _plot_labels(component, ax, scale_factor, coords):
    """Plot component labels with smart coordinate handling."""
    if not (hasattr(component, 'labels') and component.labels):
        return
        
    for label in component.labels:
        x_raw, y_raw = label.origin[0], label.origin[1]
        # Smart scaling: don't scale if coordinates are already small (likely in μm)
        x = x_raw if abs(x_raw) < 10000 else x_raw / scale_factor
        y = y_raw if abs(y_raw) < 10000 else y_raw / scale_factor
        
        coords['x'].append(x)
        coords['y'].append(y)
        
        # Modern label styling
        bbox_props = dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7", 
                         edgecolor="#F39C12", alpha=0.9)
        
        ax.text(x, y, label.text, color="#D68910", fontsize=11, ha="center", va="center",
                weight='bold', bbox=bbox_props, zorder=5)

def _plot_ports(component, ax, coords):
    """Enhanced port visualization with clustering and better aesthetics."""
    if not hasattr(component, 'ports'):
        return
        
    # Group ports by location efficiently
    port_groups = defaultdict(list)
    port_colors = ['#E74C3C', '#C0392B', '#A93226', '#922B21', '#7B241C']
    
    for name, port in (component.ports.items() if hasattr(component.ports, 'items') 
                      else enumerate(component.ports)):
        center = (port.center[0], port.center[1])
        key = (round(center[0], 3), round(center[1], 3))
        # Extract the actual port name from the port object, not the dictionary key
        actual_port_name = getattr(port, 'name', name)
        port_groups[key].append({
            'name': actual_port_name, 'center': center,
            'orientation': getattr(port, 'orientation', 0),
            'width': getattr(port, 'width', 1.0)
        })
        coords['x'].append(center[0])
        coords['y'].append(center[1])
    
    # Plot ports with enhanced styling
    for i, (location, ports_at_location) in enumerate(port_groups.items()):
        if len(ports_at_location) == 1:
            _plot_single_port(ax, ports_at_location[0], port_colors[i % len(port_colors)])
        else:
            _plot_port_cluster(ax, ports_at_location, port_colors)

def _plot_single_port(ax, port_info, color):
    """Plot a single port with direction indicator."""
    x, y = port_info['center']
    
    # Port marker - smaller and more transparent
    ax.plot(x, y, 'o', color=color, markersize=8, 
           markeredgecolor='white', markeredgewidth=1, zorder=10, alpha=0.8)
    
    # Calculate smart label position to avoid overlapping with geometry
    # Place label further away and adjust based on orientation
    orientation = port_info['orientation']
    
    # Determine offset direction based on port orientation
    if -45 <= orientation <= 45 or 315 <= orientation <= 360:  # Pointing right
        offset_x, offset_y = 40, 10
        ha, va = "left", "center"
    elif 45 < orientation <= 135:  # Pointing up
        offset_x, offset_y = 0, 40
        ha, va = "center", "bottom"
    elif 135 < orientation <= 225:  # Pointing left
        offset_x, offset_y = -40, 10
        ha, va = "right", "center"
    else:  # Pointing down
        offset_x, offset_y = 0, -40
        ha, va = "center", "top"
    
    label_x = x + offset_x
    label_y = y + offset_y
    
    # Port label with minimal, highly transparent background
    label_text = f"{port_info['name']}\n{port_info['width']:.1f}μm"
    ax.text(label_x, label_y, label_text, color=color, fontsize=10, ha=ha, va=va,
            weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
            alpha=0.3, edgecolor=color, linewidth=0.5), zorder=10)
    
    # Direction arrow - smaller
    arrow_len = 15
    dx = arrow_len * np.cos(np.radians(orientation))
    dy = arrow_len * np.sin(np.radians(orientation))
    ax.arrow(x, y, dx, dy, head_width=3, head_length=2, fc=color, ec='white', 
            alpha=0.7, linewidth=1, zorder=10)

def _plot_port_cluster(ax, ports_at_location, colors):
    """Plot overlapping ports as a cluster with radial layout."""
    center = ports_at_location[0]['center']
    x, y = center
    n_ports = len(ports_at_location)
    
    # Cluster marker - smaller and more transparent
    ax.plot(x, y, 's', color='#E74C3C', markersize=12, 
           markeredgecolor='white', markeredgewidth=2, zorder=10, alpha=0.7)
    
    # Radial port labels - placed further away
    radius = 50
    for i, port_info in enumerate(ports_at_location):
        angle = i * 360 / n_ports
        label_x = x + radius * np.cos(np.radians(angle))
        label_y = y + radius * np.sin(np.radians(angle))
        color = colors[i % len(colors)]
        
        # Port info label with minimal background
        label_text = f"{port_info['name']}\n{port_info['width']:.1f}μm"
        ax.text(label_x, label_y, label_text, color=color, fontsize=9, ha="center", va="center",
                weight='bold', bbox=dict(boxstyle="round,pad=0.2", facecolor="white", 
                alpha=0.25, edgecolor=color, linewidth=0.5), zorder=10)
        
        # Connection line - more subtle
        ax.plot([x, label_x], [y, label_y], color=color, linestyle=':', 
               alpha=0.5, linewidth=1, zorder=9)
        
        # Direction arrow - smaller
        arrow_len = 12
        dx = arrow_len * np.cos(np.radians(port_info['orientation']))
        dy = arrow_len * np.sin(np.radians(port_info['orientation']))
        ax.arrow(x, y, dx, dy, head_width=2, head_length=1.5, fc=color, ec='white', 
                alpha=0.6, linewidth=0.8, zorder=10)
    
    # Cluster label - more transparent and positioned better
    ax.text(x, y - 25, f"{n_ports} ports", color='#E74C3C', fontsize=9, 
           ha='center', va='top', weight='bold',
           bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF2F2", 
           edgecolor="#E74C3C", alpha=0.4, linewidth=0.5), zorder=10)

def _finalize_plot(ax, coords, component_name, geometry_plotted, with_ports, with_labels):
    """Apply final styling and layout to the plot."""
    ax.set_aspect("equal", "box")
    
    # Smart axis limits with padding
    if coords['x'] and coords['y']:
        x_min, x_max = min(coords['x']), max(coords['x'])
        y_min, y_max = min(coords['y']), max(coords['y'])
        
        range_x = x_max - x_min or 20
        range_y = y_max - y_min or 20
        padding = max(range_x, range_y) * 0.12
        
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
    else:
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
    
    # Modern styling
    title_color = '#2C3E50'
    ax.set_title(f"Component View: {component_name}", fontsize=16, fontweight='bold', 
                color=title_color, pad=20)
    ax.grid(True, linestyle="--", alpha=0.2, color='gray')
    ax.set_xlabel("X Position (μm)", fontsize=12, color=title_color)
    ax.set_ylabel("Y Position (μm)", fontsize=12, color=title_color)
    
    # Legend
    legend_elements = []
    if geometry_plotted:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor='#1B4F72', markersize=10, 
                                        label='Waveguides', markeredgecolor='black'))
    if with_ports:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='#E74C3C', markersize=10, 
                                        label='Ports', markeredgecolor='black'))
    if with_labels:
        legend_elements.append(plt.Line2D([0], [0], marker=None, color='#D68910', 
                                        label='Labels', linestyle='-', linewidth=3))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, 
                 fontsize=10, fancybox=True, shadow=True)
    
    # Clean modern look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')