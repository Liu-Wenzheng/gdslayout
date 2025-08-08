import numpy as np
from scipy.spatial import KDTree
import time

def distance_point_to_segment(point, seg_start, seg_end):
    """
    计算一个点到一条有限线段的最短距离和最近点坐标。
    这是一个关键的几何辅助函数。

    Args:
        point (np.array): 目标点的坐标 [x, y]。
        seg_start (np.array): 线段起点的坐标 [x, y]。
        seg_end (np.array): 线段终点的坐标 [x, y]。

    Returns:
        tuple: (最短距离, 线段上的最近点坐标)。
    """
    # 线段向量
    line_vec = seg_end - seg_start
    # 计算线段长度的平方，以避免开方运算，提高效率
    line_len_sq = np.sum(line_vec**2)
    
    # 特殊情况：如果线段的起点和终点重合，它是一个点
    if line_len_sq < 1e-9:
        return np.linalg.norm(point - seg_start), seg_start

    # 将点投影到线段所在的无限长直线上。
    # t 是投影点在线段方向上的相对位置。
    t = np.dot(point - seg_start, line_vec) / line_len_sq
    
    # 根据t的值判断最近点的位置
    if t < 0.0:
        closest_point = seg_start
    elif t > 1.0:
        closest_point = seg_end
    else:
        closest_point = seg_start + t * line_vec
        
    distance = np.linalg.norm(point - closest_point)
    return distance, closest_point

def find_shortest_distances_kdtree(curve_points, points_to_test, k=5):
    """
    【核心函数】
    使用 scipy.spatial.KDTree 高效计算点集到分段线性曲线的【带符号】最短距离和【距离向量】。

    Args:
        curve_points (np.array): 定义曲线的N*2点集, 必须按顺序连接。
        points_to_test (np.array): 待测的M*2点集。
        k (int): 对于每个待测点，查询k个最近的曲线顶点以确定候选线段。

    Returns:
        tuple: (带符号的最短距离数组, 对应的距离向量数组)
    """
    # print(f"开始计算... 曲线点数: {len(curve_points)}, 待测点数: {len(points_to_test)}")
    # start_time = time.time()
    
    curve_kdtree = KDTree(curve_points)
    segments = np.array(list(zip(curve_points[:-1], curve_points[1:])))
    
    signed_distances = []
    distance_vectors = [] # <--- 修改：存储距离向量
    
    for pt in points_to_test:
        _, indices = curve_kdtree.query(pt, k=k)
        
        min_dist_for_pt = np.inf
        closest_pt_for_pt = None
        best_seg_idx = -1

        candidate_segment_indices = set()
        for idx in indices:
            if idx > 0:
                candidate_segment_indices.add(idx - 1)
            if idx < len(curve_points) - 1:
                candidate_segment_indices.add(idx)

        if not candidate_segment_indices:
             candidate_segment_indices.add(indices[0] if indices[0] < len(segments) else indices[0]-1)

        for seg_idx in candidate_segment_indices:
            seg_start, seg_end = segments[seg_idx]
            dist, closest_pt_on_seg = distance_point_to_segment(pt, seg_start, seg_end)
            
            if dist < min_dist_for_pt:
                min_dist_for_pt = dist
                closest_pt_for_pt = closest_pt_on_seg
                best_seg_idx = seg_idx
        
        # --- 计算距离的符号 ---
        sign = 0.0
        vec_to_point = pt - closest_pt_for_pt # <--- 修改：计算向量
        
        if best_seg_idx != -1 and min_dist_for_pt > 1e-9:
            seg_start, seg_end = segments[best_seg_idx]
            tangent = seg_end - seg_start
            normal = np.array([-tangent[1], tangent[0]])
            dot_product = np.dot(vec_to_point, normal)
            sign = np.sign(dot_product)

        signed_distances.append(min_dist_for_pt * sign)
        distance_vectors.append(vec_to_point) # <--- 修改：存储向量
    
    # end_time = time.time()
    # print(f"计算完成，耗时: {end_time - start_time:.4f} 秒。")

    return np.array(signed_distances), np.array(distance_vectors)


class Ring_down_distance:
    def __init__(self, component1, component2, shift1=(0, 0), shift2=(0, 0), layer=(2, 0)):
        self.component1 = component1
        self.component2 = component2
        self.shift1 = shift1
        self.shift2 = shift2
        self.layer = layer

    def get_points_from_component(self, component, boundary=['r']):
        region = component.get_region(layer=self.layer)
        bbox = region.bbox()
        # Process boundary parameter to determine which edges of the bounding box to include
        bpoints = []
        if boundary is None or not boundary:
            bpoints = []  # No boundary points included
        elif isinstance(boundary, list):
            # Map each direction code to corresponding bbox boundary point(s)
            mapping = {
                'l': [bbox.left],
                'r': [bbox.right],
                't': [bbox.top],
                'b': [bbox.bottom],
                'all': [bbox.left, bbox.right, bbox.top, bbox.bottom]
            }
            
            # Collect boundary points based on specified directions
            for direction in boundary:
                if direction in mapping:
                    bpoints.extend(mapping[direction])
                elif direction == 'all':
                    bpoints = list(mapping['all'])
                    break
        else:
            # If boundary is a single string, handle it directly
            if boundary == 'all':
                bpoints = [bbox.left, bbox.right, bbox.top, bbox.bottom]
            elif boundary in ['l', 'r', 't', 'b']:
                mapping = {'l': bbox.left, 'r': bbox.right, 't': bbox.top, 'b': bbox.bottom}
                bpoints = [mapping[boundary]]
        effective_regions = []
        for i, polygon_with_props in enumerate(region):
            for point in polygon_with_props.each_point_hull():
                if point.x in bpoints or point.y in bpoints:
                    effective_regions.append(polygon_with_props)
                    break
        effective_points = []
        for polygon in effective_regions:
            for point in polygon.each_point_hull():
                effective_points.append([point.x, point.y])
        effective_points = np.array(effective_points)
        return effective_regions, effective_points

    def calculate(self, type='exp', alpha=1.0):
        region1, points1 = self.get_points_from_component(self.component1.copy().move(self.shift1), boundary=['r'])
        bbox1 = region1[0].bbox()
        bpoints_start = [bbox1.left, bbox1.top]
        idx_start = np.argmin(np.linalg.norm(points1 - bpoints_start, axis=1))
        bpoint_end = [bbox1.right, bbox1.bottom]
        idx_end = np.argmin(np.linalg.norm(points1 - bpoint_end, axis=1))
        points1_path = points1[idx_start:idx_end+1]
        points2 = self.get_points_from_component(self.component2.copy().move(self.shift2), boundary=['l', 'b'])[1]

        distance, vector = find_shortest_distances_kdtree(points1_path, points2, k=5)

        if type == 'exp':
            result = self.weighted_average_exp(vector.T, distance, k=alpha)
        elif type == 'linear':
            result = self.weighted_average_linear(vector.T, distance, alpha=alpha)
        elif type == 'min':
            result = self.weighted_average_min(vector.T, distance)
        else:
            raise ValueError("Unsupported type. Use 'exp', 'linear', or 'min'.")

        return result * 1e-3

    def weighted_average_linear(self, V, w, alpha=1.0):
        """
        线性权重:
            w' = ((w_max - w) / (w_max - w_min)) ** alpha
        α≥1 时，α 越大，越聚焦到最小 (最负) 的 w。
        """
        w = np.asarray(w, dtype=float)
        V = np.asarray(V, dtype=float)
        w_max, w_min = w.max(), w.min()
        if w_max == w_min:                 # 退化为普通平均
            weights = np.ones_like(w) / len(w)
        else:
            weights_raw = (w_max - w) / (w_max - w_min)
            weights_raw = weights_raw**alpha
            weights = weights_raw / weights_raw.sum()
        return V @ weights                # 2‑维向量

    def weighted_average_exp(self, V, w, k=1.0, method="shift"):
        V = np.asarray(V, dtype=float)
        w = np.asarray(w, dtype=float)

        if method == "shift":
            z = -k * w                     # larger for more‑negative w
            z -= z.max()                  # shift so max(z)=0 ⇒ exp ≤ 1
            weights_raw = np.exp(z)
        elif method == "linear":
            w_max, w_min = w.max(), w.min()
            if w_max == w_min:
                weights_raw = np.ones_like(w)
            else:
                mapped = (w_max - w) / (w_max - w_min)   # ∈[0,1]
                weights_raw = np.exp(-k * mapped)
        else:
            raise ValueError("method must be 'shift' or 'linear'")

        weights = weights_raw / weights_raw.sum()
        return V @ weights

    def weighted_average_min(self, V, w):
        """直接返回对应最负 w 的列。"""
        idx = int(np.argmin(w))
        return V[:, idx]
