"""
算法工具。

该工具实现了一些常用的算法。这些算法可以使用 Numba 进行加速。
"""

from ..externals import np, plt
# from scipy.spatial import Voronoi, voronoi_plot_2d
from numba import njit, prange


@njit(parallel=True)
def kmeans_numba(points, num_clusters, max_iter=100):
    """
    使用 Numba 实现的 K-means 算法。

    Args:
        points (ndarray): 二维数组，每一行代表一个点。
        num_clusters (int): 聚类的数量。
        max_iter (int): 最大迭代次数。

    Returns:
        ndarray: 二维数组，每一行代表一个质心。

    Examples:
        >>> points = np.array([[1, 2], [3, 4], [5, 6]])
        >>> kmeans_numba(points, 2)
    """

    num_points, num_dim = points.shape
    centroids = points[np.random.choice(num_points, num_clusters, replace=False)]

    for _ in range(max_iter):
        distances = np.empty((num_points, num_clusters))
        for i in prange(num_points):
            for j in prange(num_clusters):
                distances[i, j] = np.sum((points[i] - centroids[j]) ** 2)

        labels = np.argmin(distances, axis=1)

        new_centroids = np.empty((num_clusters, num_dim))
        for j in prange(num_clusters):
            cluster_points = points[labels == j]
            for d in prange(num_dim):
                new_centroids[j, d] = np.sum(cluster_points[:, d]) / cluster_points.shape[0]

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids


def find_nearest_point(target_point, points):
    """
    寻找距离目标点最近的点。

    Args:
        target_point (tuple): 目标点坐标。
        points (list): 其他点的坐标列表。

    Returns:
        nearest_point_index (int): 最近点在列表中的索引。
        min_distance (float): 最近点与目标点之间的距离。
    """
    min_distance = float('inf')
    nearest_point_index = -1
    for index, point in enumerate(points):
        distance = np.linalg.norm(np.array(point) - np.array(target_point))
        if distance < min_distance:
            min_distance = distance
            nearest_point_index = index
    return nearest_point_index, min_distance


# def generate_network_with_Voronoi(nodes_pos):
#     """
#     使用 Voronoi 图生成网络。
#
#     Args:
#         nodes_pos (np.ndarray): 节点坐标
#
#     Returns:
#         edges (np.ndarray): 边
#     """
#     # 使用 Voronoi 图
#     vor = Voronoi(nodes_pos)
#
#     # 获取边
#     edges = []
#     for ridge in vor.ridge_vertices:
#         if -1 not in ridge:
#             edges.append(ridge)
#
#     # 可视化 Voronoi 图
#     voronoi_plot_2d(vor)
#     plt.show()
#
#     return np.array(edges)


if __name__ == '__main__':

    ## 示例展示 K-means 算法
    import matplotlib.pyplot as plt
    import timeit

    # 创建一些随机数据
    time_010 = timeit.default_timer()
    num_points = 10000
    points = np.empty((num_points, 2))
    for i in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1))
        points[i] = [r * np.cos(theta), r * np.sin(theta)]

    # 使用 K-means 算法生成 100 个散点
    num_clusters = 100
    centroids = kmeans_numba(points, num_clusters)
    time_020 = timeit.default_timer()
    print('运行用时：', time_020 - time_010)

    # 绘制结果
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()
