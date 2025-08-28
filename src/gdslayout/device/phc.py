import math
from typing import List, Iterable, Union, Dict

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import Any
from . import register_device


class HexGrid:
    """
    A class to create, manage, and transform hexagonal grids.
    """

    # Initialization & Grid Generation
    def __init__(self, grid_x_range: Iterable[int], grid_y_range: Iterable[int],
                 spacing: float = 10.0, offset_first_row: bool = False):
        self._sorted_grid_x_range = sorted(list(grid_x_range))
        self._sorted_grid_y_range = sorted(list(grid_y_range))
        self.spacing = spacing
        self.offset_first_row = offset_first_row

        self._all_points_data_2d: List[List[Dict]] = self._generate_all_points_data_2d()
        self._all_points_data_flat: List[Dict] = [point for row in self._all_points_data_2d for point in row]
        self._center_hexes = self._find_center_hexes()

    def _generate_all_points_data_2d(self) -> List[List[Dict]]:
        """
        Generates a 2D list of dictionaries, each containing cartesian (as np.array)
        and cube coordinates, and assigns a unique point_id.
        """
        points_data_2d = []
        sqrt3 = math.sqrt(3)
        offset_logic = 1 if self.offset_first_row else 0
        point_id_counter = 0

        for row_val in self._sorted_grid_y_range:
            current_row_points = []
            for col_val in self._sorted_grid_x_range:
                if not self.offset_first_row:
                    q = col_val - (row_val - (row_val & 1)) // 2
                else:
                    q = col_val - (row_val + (row_val & 1)) // 2
                r = row_val
                offset = 0.5 * ((row_val + offset_logic) % 2)
                x = self.spacing * (col_val + offset)
                y = self.spacing * row_val * sqrt3 / 2

                point_data = {
                    'point_id': point_id_counter,
                    'cartesian': np.array([x, y], dtype=float),
                    'cube': (q, -q - r, r),
                    'grid_row': row_val,
                    'grid_col': col_val
                }
                current_row_points.append(point_data)
                point_id_counter += 1
            points_data_2d.append(current_row_points)
        return points_data_2d

    # Core Grid Properties
    @staticmethod
    def hex_distance(c1: tuple, c2: tuple) -> int:
        """Calculates the hexagonal distance between two cube coordinates."""
        return (abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])) // 2

    def _find_center_hexes(self) -> List[tuple]:
        """Finds the center hex(es) for the current grid configuration."""
        if not self._all_points_data_flat: return []
        center_hexes_cube = []
        if not self.offset_first_row:
            center_point_data = min(self._all_points_data_flat, key=lambda p: self.hex_distance(p['cube'], (0, 0, 0)))
            center_hexes_cube.append(center_point_data['cube'])
        else:
            if len(self._all_points_data_flat) < 2:
                center_hexes_cube = [p['cube'] for p in self._all_points_data_flat]
            else:
                # cartesian coordinates are now np.array, access using [0] and [1] still works
                sorted_points = sorted(self._all_points_data_flat, key=lambda p: p['cartesian'][0]**2 + p['cartesian'][1]**2)
                center_hexes_cube.append(sorted_points[0]['cube'])
                center_hexes_cube.append(sorted_points[1]['cube'])
        return center_hexes_cube

    def get_max_group_id(self) -> int:
        """Calculates the maximum group ID (max distance from the center)."""
        if not self._center_hexes: return -1
        max_dist = 0
        for point_data in self._all_points_data_flat:
            dist = min(self.hex_distance(point_data['cube'], center_hex) for center_hex in self._center_hexes)
            if dist > max_dist: max_dist = dist
        return max_dist

    # Point Access & Counting
    def get_points(self) -> np.ndarray:
        """
        Returns a 2D NumPy array of the current Cartesian grid points (N, 2).
        Each row represents a point (x, y).
        """
        if not self._all_points_data_flat:
            return np.empty((0, 2), dtype=float)
        return np.array([p['cartesian'] for p in self._all_points_data_flat], dtype=float)

    def get_indexed_points(self) -> List[Dict]:
        """
        Returns a flat list of all points, each augmented with a 'point_id' for indexing.
        This allows external functions to easily reference points by their unique ID.
        """
        return self._all_points_data_flat

    def num_per_row(self) -> Dict[int, int]:
        """
        Returns the number of points for each row.
        The keys are the original grid_y_range values (row identifiers).
        """
        row_counts = {}
        for row_points in self._all_points_data_2d:
            if row_points:
                row_id = row_points[0]['grid_row']
                row_counts[row_id] = len(row_points)
        return row_counts

    def num_per_col(self) -> Dict[int, int]:
        """
        Returns the number of points for each column.
        The keys are the original grid_x_range values (column identifiers).
        """
        col_counts = {col_val: 0 for col_val in self._sorted_grid_x_range}
        for row_points in self._all_points_data_2d:
            for point_data in row_points:
                col_id = point_data['grid_col']
                col_counts[col_id] += 1
        return col_counts

    # Point Manipulation
    def _get_assignments(self, vector: List[Union[int, float]]) -> List[Union[int, float]]:
        """Calculates translation values for each point based on its group and a vector."""
        if not self._center_hexes: return [0] * len(self._all_points_data_flat)
        assignments = []
        for point_data in self._all_points_data_flat:
            group_id = min(self.hex_distance(point_data['cube'], center_hex) for center_hex in self._center_hexes)
            value = vector[group_id] if group_id < len(vector) else 0
            assignments.append(value)
        return assignments

    def apply_translation_to_points(self, vector: List[Union[int, float]]):
        """
        Applies vertical translation to the internal Cartesian coordinates of all points.
        The changes are permanent and will be reflected by subsequent calls to get_points().
        """
        assignments = self._get_assignments(vector)
        for i, point_data in enumerate(self._all_points_data_flat):
            # Directly modify the y-coordinate of the NumPy array stored in 'cartesian'
            point_data['cartesian'][1] += assignments[i]


    def remove_point_by_id(self, point_ids_to_remove: Union[int, List[int]]):
        """
        Removes point(s) with the given point_id(s) from the grid.
        Accepts either a single integer point_id or a list of point_ids.
        Updates both the 2D and flat point data structures.
        """
        if isinstance(point_ids_to_remove, int):
            ids_to_remove_set = {point_ids_to_remove}
        else:
            ids_to_remove_set = set(point_ids_to_remove)

        if not ids_to_remove_set:
            print("No point IDs provided for removal.")
            return
        
        removed_count = 0

        new_flat_points = []
        for point in self._all_points_data_flat:
            if point['point_id'] not in ids_to_remove_set:
                new_flat_points.append(point)
            else:
                removed_count += 1

        self._all_points_data_flat = new_flat_points

        if removed_count == 0:
            print(f"Warning: None of the provided point IDs ({list(ids_to_remove_set)}) were found.")
        elif removed_count < len(ids_to_remove_set):
            found_ids_in_new_flat = {point['point_id'] for point in self._all_points_data_flat}
            actually_removed_ids = ids_to_remove_set.difference(found_ids_in_new_flat)
            not_found_ids = ids_to_remove_set.difference(actually_removed_ids)
            if not_found_ids:
                print(f"Warning: Some provided IDs were not found: {list(not_found_ids)}.")

        # Reconstruct the 2D list based on the updated flat list
        new_all_points_data_2d = []
        points_by_row_val = {row_val: [] for row_val in self._sorted_grid_y_range}
        for point in self._all_points_data_flat:
            points_by_row_val[point['grid_row']].append(point)

        for row_val in self._sorted_grid_y_range:
            new_all_points_data_2d.append(
                sorted(points_by_row_val[row_val], key=lambda p: p['grid_col'])
            )
        self._all_points_data_2d = new_all_points_data_2d

        # Recalculate center hexes as the grid configuration might have changed
        self._center_hexes = self._find_center_hexes()


@register_device("PhC_optomechanics1")
def PhC_optomechanics1(
    length: float = 100, 
    half_height: float = 50, 
    phc_width: float = 6.0, 
    spacing: float = 4.0, 
    radius: float = 1.0, 
    center_displacement_offset: List[float] = [2, 2, 2], 
    side_displacement_offset: float = 2, 
    side_order: int = 8,
    bbox_cut: bool = False,
    resolution: float = 0.02, 
    layer: Any = (1,0),
) -> gf.Component:
    """
    Creates a Photonic Crystal Type 1
    """

    c = gf.Component()
    angle_res = 360 * resolution / (2 * np.pi * radius)
    generic_hole = gf.components.circle(radius=radius, angle_resolution=angle_res, layer=layer)

    x_num = int(length / spacing / 2) + int(radius / spacing) + 2
    y_num = int(half_height / spacing) + int(radius / spacing) + 2
    grid_x = range(-x_num, x_num + 1)
    grid_y = range(0, y_num)

    hexgrid = HexGrid(grid_x, grid_y, spacing=spacing, offset_first_row=True)
    max_id = hexgrid.get_max_group_id()

    displacement_offset = []
    if side_order and max_id > 0 and max_id >= side_order:
        num_core_offsets = len(center_displacement_offset)
        vector_len = max_id + 1
        displacement_offset.extend(center_displacement_offset)
        displacement_offset.extend([0] * (side_order - num_core_offsets))
        displacement_offset.extend([side_displacement_offset] * (vector_len - len(displacement_offset)))
    else:
        displacement_offset = center_displacement_offset

    hexgrid.apply_translation_to_points(displacement_offset)
    points_field = hexgrid.get_points()
    points_field[:, 1] += phc_width / 2 + radius
    points_field = np.concatenate((points_field, points_field * np.array([1, -1])), axis=0)

    # Bounding Box and Polygon Collection Logic
    bbox = (-length/2, -half_height, length/2, half_height)
    polygon_list = []
    
    if not bbox_cut:
        xmin, ymin, xmax, ymax = bbox
        all_holes_fully_contained = gf.Component()

        for center in points_field:
            if (center[0] - radius >= xmin) and (center[0] + radius <= xmax):
                all_holes_fully_contained.add_ref(generic_hole).move(center)

        c.add_ref(all_holes_fully_contained)

        for ly_tuple, polys in all_holes_fully_contained.get_polygons_points(by='tuple').items():
            if ly_tuple == tuple(layer):
                polygon_list.extend([np.array(poly) for poly in polys])
            
    else:
        all_holes = gf.Component()
        for center in points_field:
            if (abs(center[0]) < length/2 + radius) and (abs(center[1]) < half_height + radius):
                all_holes.add_ref(generic_hole).move(center)
        
        bbox_comp = gf.components.rectangle(size=(length, 2*half_height), centered=True, layer=layer)
        clipped_holes_component = gf.boolean(A=all_holes, B=bbox_comp, operation='and', layer=layer)

        polygon_list = clipped_holes_component.get_polygons_points(by='tuple')[tuple(layer)]
        c.add_ref(clipped_holes_component)

    c.add_port(name="bottom", center=(0, c.ymin), width=1, orientation=270, layer=layer)
    c.add_port(name="coupler", center=(0, 0), width=1, orientation=0, layer=layer)

    return c, polygon_list
    

@register_device("PhC_optomechanics2")
def PhC_optomechanics2(
    length: float = 100, 
    height_num1: int = 10, 
    height_num2: int = 1, 
    height_num3: float = 3,
    height_num4: int = 10,
    spacing: float = 4.0, 
    radius: float = 1.0, 
    coupling_phc_num: int = 6,
    phc_width: float = 6.0,
    slip_width: float = 1.0,
    center_displacement_offset: List[float] = [2, 2, 2], 
    side_displacement_offset: float = 2, 
    side_order: int = 8,
    bbox_cut: bool = False,
    resolution: float = 0.02, 
    layer: Any = (1,0),
) -> gf.Component:
    """
    Creates a Photonic Crystal Type 2
    """

    c = gf.Component()
    angle_res = 360 * resolution / (2 * np.pi * radius)
    generic_hole = gf.components.circle(radius=radius, angle_resolution=angle_res, layer=layer)

    x_num = int(length / spacing / 2) + int(radius / spacing) + 2
    grid_x = range(-x_num, x_num + 1)

    points_field_list = []
    for i, grid_y in enumerate([range(0, height_num1 + height_num2 + height_num3), range(0, height_num4)]):
        hexgrid = HexGrid(grid_x, grid_y, spacing=spacing, offset_first_row=True)
        max_id = hexgrid.get_max_group_id()

        displacement_offset = []
        if side_order and max_id > 0 and max_id >= side_order:
            num_core_offsets = len(center_displacement_offset)
            vector_len = max_id + 1
            displacement_offset.extend(center_displacement_offset)
            displacement_offset.extend([0] * (side_order - num_core_offsets))
            displacement_offset.extend([side_displacement_offset] * (vector_len - len(displacement_offset)))
        else:
            displacement_offset = center_displacement_offset

        if i == 0:
            x_number = 2 * x_num + 1
            k0 = np.sum(x_number*height_num3)
            array1 = np.arange(k0, k0+x_number, 1)
            array2v= np.arange(k0 + (x_number-coupling_phc_num)//2, k0 + (x_number+coupling_phc_num)//2, 1)
            point_r_index = np.setdiff1d(array1, array2v)
            hexgrid.remove_point_by_id(point_r_index)
        hexgrid.apply_translation_to_points(displacement_offset)
        points_field = hexgrid.get_points()
        points_field[:, 1] += phc_width / 2 + radius
        points_field_list.append(points_field)

    points_field = np.concatenate((points_field_list[0], points_field_list[1] * np.array([1, -1])), axis=0)

    # Bounding Box and Polygon Collection Logic
    bbox_x = (-length/2, length/2)
    polygon_list = []
    
    if not bbox_cut:
        xmin, xmax = bbox_x
        all_holes_fully_contained = gf.Component()

        for center in points_field:
            if (center[0] - radius >= xmin) and (center[0] + radius <= xmax):
                all_holes_fully_contained.add_ref(generic_hole).move(center)

        c.add_ref(all_holes_fully_contained)

        for ly_tuple, polys in all_holes_fully_contained.get_polygons_points(by='tuple').items():
            if ly_tuple == tuple(layer): # Only collect polygons on the specified layer
                polygon_list.extend([np.array(poly) for poly in polys])
            
    else:
        all_holes = gf.Component()
        for center in points_field:
            if (abs(center[0]) < length/2 + radius):
                all_holes.add_ref(generic_hole).move(center)
        
        bbox_comp = gf.components.rectangle(size=(length, all_holes.ysize), centered=True, layer=layer)
        all_holes_center = all_holes.center
        all_holes.move(all_holes_center, bbox_comp.center)
        clipped_holes_component = gf.boolean(A=all_holes, B=bbox_comp, operation='and', layer=layer)
        clipped_holes_component.move(clipped_holes_component.center, all_holes_center)

        polygon_list = clipped_holes_component.get_polygons_points(by='tuple')[tuple(layer)]
        c.add_ref(clipped_holes_component)

    slip = gf.components.rectangle(size=(length, slip_width), centered=True, layer=layer)
    c.add_ref(slip)
    coupling_port = height_num3 * spacing * np.sqrt(3)/2 + radius + phc_width / 2 + side_displacement_offset
    c.add_port(name="top", center=(0, c.ymax), width=1, orientation=90, layer=layer)
    c.add_port(name="bottom", center=(0, c.ymin), width=1, orientation=270, layer=layer)
    c.add_port(name="in", center=(-length/2, coupling_port), width=1, orientation=180, layer=layer)
    c.add_port(name="out", center=(length/2, coupling_port), width=1, orientation=0, layer=layer)
    c.add_port(name="coupler", center=(0, coupling_port), width=1, orientation=0, layer=layer)
    c.add_port_with_info(name='exciton', center=(0, 0), width=1, orientation=0, layer=layer, info={'exciton_width': float((1 + 2*len(center_displacement_offset))*spacing) - 2*radius, 'exciton_height': phc_width})
    return c, polygon_list