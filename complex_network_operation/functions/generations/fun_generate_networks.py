"""
函数：生成网络功能
"""
import networkx as nx

from complex_network_operation.externals import np, plt
# from complex_network_operation.externals import njit
from complex_network_operation.functions.generations.fun_generate_points import generate_points
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib
# matplotlib.use('TkAgg')


def generate_network(nodes_pos: np.ndarray, graph_direction_type='undirected', set_num_edges: int = None, set_num_interpolated_density_distance: float = None, num_edges_per_node: int = None, num_neighbors: int = None, network_machanism: str = 'Voronoi', network_type: str = 'Voronoi', hierarchical: bool = False, is_preview_plot: bool = False):
    """
    生成不同机制的、不同网络连接结构的，具有 NetworkX 工具包图数据类型的图网络。

    这个方法的步骤是：
    1. 获取节点之坐标；
    2. 根据网络生成机制生成网络之连边；

    - 网络连接结构 network_type 有以下可选项：#HACK 现在只做了加权的 Voronoi 图生成网络
        - 'Random': 随机生成网络
        - 'Voronoi': 使用 Voronoi 图生成网络；
        - 'Complete': 使用完全图生成网络；
        - 'Random': 使用随机图生成网络；
        - 'Community Structure': 使用社区结构生成网络；
        - 'Small World': 使用小世界网络生成网络；
        - 'Scale Free': 使用无标度网络生成网络；
        - 'Hierarchical': 使用分层网络生成网络；
        - 'Regular': 使用规则网络生成网络；
        - 'Grid': 使用网格网络生成网络；
        - 'Scale Free': 使用无标度网络生成网络；

    Args:
        nodes_pos (np.ndarray): 节点坐标
        graph_direction_type (str): 图的方向类型。默认是 'undirected' 。可选值有：
            - 'directed': 有向图
            - 'undirected': 无向图
        set_num_edges (int): 设置边数量
        num_edges_per_node (int): 每个节点的边数量
        num_neighbors (int): 邻居节点数量
        network_machanism (str): 网络生成机制。默认是 Voronoi 图。
        network_type (str): 网络连接结构。默认是 Voronoi 图。
        hierarchical (bool): 是否生成分层网络。默认是 False。
        is_preview_plot (bool): 是否预览绘制。默认是 False。

    Returns:
        图网络。图网络有两种可能的类型：有向图或者无向图。

        图网络数据结构：

        - graph: 图属性
        - g.nodes: 节点
          - id: 节点 ID
            - name: 节点名称。默认是节点 ID 字符串
            - type: 节点类型。默认 'node'
            - pos: 节点坐标
            - color_name: 节点颜色名称。默认 'red'
        - g.edges: 边
            - id: 边 ID
            - name: 边名称。默认是 'edge:{node1}-{node2}'
            - type: 边类型。默认 'edge'
            - color_name: 边颜色名称。默认 'black'
            - straight_distance: 直线距离


        vor (Voronoi): Voronoi 类数据。包含多个属性，如 vertices, ridge_points, ridge_vertices, regions, point_region, furthest_site 等。

    """

    match graph_direction_type:
        case 'undirected':
            g = nx.Graph()
        case 'directed':
            g = nx.DiGraph()
            pass  # match

    # 添加节点
    for i, node_pos in enumerate(nodes_pos):
        g.add_node(
            i,
            id=i,
            name=str(i),
            type='node',
            pos=node_pos.copy(),
            color_name='red',
        )

    # TODO 调用不同的网络生成机制
    match network_type:
        case 'Voronoi':
            edges, vor = _generate_weighted_voronoi_network(nodes_pos, weights=None, is_preview_plot=is_preview_plot)

        case 'Hierarchical':
            _generate_hierarchical_network(nodes_pos, set_num_edges, num_edges_per_node, num_neighbors)  # #TODO 没开发，不能使用

        case _:
            errors = f"network_type {network_type} is not supported."
            raise ValueError(errors)
            pass  # match

    # 添加边
    match graph_direction_type:
        case 'undirected':
            for i, edge in enumerate(edges):
                node1, node2 = edge
                g.add_edge(
                    node1,
                    node2,
                    id=i,
                    name=f"edge:{node1}-{node2}",
                    type='edge',
                    color_name='black',
                    straight_distance=np.linalg.norm(vor.points[node1] - vor.points[node2]),
                )
                pass  # for
        case 'directed':
            for i, edge in enumerate(edges):
                node1, node2 = edge
                g.add_edge(
                    node1,
                    node2,
                    id=i,
                    name=f"edge:{node1}-{node2}",
                    type='edge',
                    color_name='black',
                    straight_distance=np.linalg.norm(vor.points[node1] - vor.points[node2]),
                )
                g.add_edge(
                    node2,
                    node1,
                    id=i + len(edges),  # Ensure unique id for the reverse edge
                    name=f"edge:{node2}-{node1}",
                    type='edge',
                    color_name='black',
                    straight_distance=np.linalg.norm(vor.points[node2] - vor.points[node1]),
                )
                pass  # for
            pass  # match

    return g, vor
    pass  # function


# NOTE 使用普通的邻居节点生成网络

def generate_neighbors_network(nodes_pos, num_edges_per_node, num_neighbors):
    """
    生成图网络。

    这个方法的步骤是：
    1. 获取节点之坐标；
    2. 计算节点之间的距离；
    3. 选择邻居节点；
    4. 生成边；

    Args:
        nodes_pos (np.ndarray): 节点坐标
        set_num_edges (int): 设置边数量
        num_edges_per_node (int): 每个节点的边数量
        num_neighbors (int): 邻居节点数量

    Returns:
        edges (np.ndarray): 边

    """
    distance_matrix = _calculate_distance_matrix(nodes_pos)  # 计算节点之间的距离
    neighbors = _select_neighbors(distance_matrix, len(nodes_pos), num_neighbors)  # 选择邻居节点
    edges = _create_edges(nodes_pos, neighbors, num_edges_per_node)  # 生成边

    return edges
    pass  # function


# @njit
def _calculate_distance_matrix(nodes, type_of_distance='Euclidean', is_sparse=False):
    """
    计算节点之间的距离矩阵。

    计算原理是：对于每一个节点，计算它与其他节点之间的直线距离。

    Args:
        nodes (np.ndarray): 节点坐标
        type_of_distance (str): 距离类型。默认是欧几里得距离。可选值有：
            - 'Euclidean': 欧几里得距离
            - 'Manhattan': 曼哈顿距离  #TODO 没开发，不能使用
            - 'Chebyshev': 切比雪夫距离  #TODO 没开发，不能使用
            - 'Sphaerical': 球面距离  #TODO 没开发，不能使用

        is_sparse (bool): 是否表示为稀疏矩阵。默认是 False。

    Returns:
        np.array: 距离矩阵。是一个二维数组，每个元素表示两个节点之间的直线距离。

    """
    num_nodes = nodes.shape[0]
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance_matrix[i, j] = np.linalg.norm(nodes[i] - nodes[j])
            distance_matrix[j, i] = distance_matrix[i, j]  # 对称矩阵
    return distance_matrix
    pass  # function


# @njit
def _select_neighbors(distance_matrix, num_nodes, num_neighbors):
    """
    根据节点之间的距离选择邻居节点

    Args:
        distance_matrix (np.ndarray): 距离矩阵
        num_nodes (int): 节点数量
        num_neighbors (int): 邻居节点数量

    Returns:
        neighbors: list, 邻居节点索引
    """
    neighbors = []
    for i in range(num_nodes):
        sorted_indices = np.argsort(distance_matrix[i])
        neighbors.append(sorted_indices[1:num_neighbors + 1])  # 排除自己，选择距离最近的 num_neighbors 个节点
    return neighbors
    pass  # function


# 根据邻居节点生成边


# @njit
def _create_edges(nodes, neighbors, num_edges):
    num_nodes = nodes.shape[0]
    edges = []
    for i in range(num_nodes):
        for _ in range(num_edges):
            neighbor_index = np.random.choice(neighbors[i])
            edges.append((i, neighbor_index))
    return np.array(edges)
    pass  # function


# @njit
def _calculate_weighted_distances(nodes_pos, weights):
    """
    计算节点之间的加权距离。

    Args:
        nodes_pos (np.ndarray): 节点坐标
        weights (np.ndarray): 权重

    Returns:
        np.array: 加权距离矩阵
    """
    num_nodes = nodes_pos.shape[0]
    weighted_distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(nodes_pos[i] - nodes_pos[j])
            weighted_distance = distance / weights[i]
            weighted_distances[i, j] = weighted_distance
            weighted_distances[j, i] = weighted_distance  # 对称矩阵
    return weighted_distances
    pass  # function


# NOTE 使用 Voronoi 图生成网络

def _generate_weighted_voronoi_network(
        nodes_pos: np.ndarray,
        weights: np.ndarray = None,
        **kwargs
):
    """
    生成加权的 Voronoi 图网络。#TODO 现在生成的仅仅是非加权的 Voronoi 图。后续再增加加权功能

    Args:
        nodes_pos (np.ndarray): 节点坐标
        weights (np.ndarray): 权重

    Returns:
        edges (np.ndarray): 边集合，每个元素是一个边，是一个二元组，表示两个节点的索引
        vor (Voronoi): Voronoi 类数据。包含多个属性，如 vertices, ridge_points, ridge_vertices, regions, point_region, furthest_site 等。

    """
    # 默认加权都是 1
    if weights is None:
        weights = np.ones(nodes_pos.shape[0])

    vor = Voronoi(nodes_pos)

    # 绘制图预览
    if kwargs['is_preview_plot']:
        # #DEBUG Plot Voronoi diagram
        fig, ax = plt.subplots()
        voronoi_plot_2d(vor, ax=ax)

        # Colorize Voronoi cells according to weights
        for region, weight in zip(vor.regions, weights):
            if not -1 in region:
                polygon = [vor.vertices[i] for i in region]
                ax.fill(*zip(*polygon), color=plt.cm.viridis(weight))

        # 显示输入点
        ax.plot(nodes_pos[:, 0], nodes_pos[:, 1], 'ro')

        # 显示 Voronoi 顶点
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'bo')

        # 显示输入点之间的连边
        for edge in vor.ridge_points:
            point1 = nodes_pos[edge[0]]
            point2 = nodes_pos[edge[1]]
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'y-')

        plt.draw()
        plt.show()

    # 根据 vor 图将节点连接起来
    edges = vor.ridge_points
    # # 简单排序边的顺序
    # edges = np.sort(edges, axis=0)

    return edges, vor
    pass  # function


def _generate_hierarchical_network(nodes_pos, set_num_edges, num_edges_per_node, num_neighbors):
    """
    生成层级网络。

    Args:
        nodes_pos (np.ndarray): 节点坐标
        set_num_edges (int): 设置边数量
        num_edges_per_node (int): 每个节点的边数量
        num_neighbors (int): 邻居节点数量

    Returns:
        edges (np.ndarray): 边
    """
    # 生成顶层网络
    top_level_edges = generate_neighbors_network(nodes_pos, num_edges_per_node, num_neighbors)

    # 生成每个城市的内部网络
    city_networks = []
    for city_pos, neighbors in zip(nodes_pos, top_level_edges):
        num_external_nodes = len(neighbors)
        city_points = generate_points(set_densityDistance=None, set_numPoints=50, set_circleRadius=10, circle_origin=city_pos, hierarchical_network_type=True, num_external_nodes=num_external_nodes)
        city_edges = generate_neighbors_network(city_points, num_edges_per_node, num_neighbors)
        city_networks.append((city_points, city_edges))

    return top_level_edges, city_networks
