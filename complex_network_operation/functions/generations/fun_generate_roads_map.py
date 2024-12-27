"""
函数：生成路网地图。
"""
from complex_network_operation.functions.generations.fun_generate_networks import generate_network
from complex_network_operation.externals import np, Path
from complex_network_operation.functions.generations.fun_generate_networks import _generate_weighted_voronoi_network, _calculate_distance_matrix
from complex_network_operation.utils.altorithm_utils import find_nearest_point

from skimage import io, color, feature, filters, morphology, measure
from skimage.feature import corner_harris
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from skimage import morphology, measure


# %% #NOTE 通过生成机制的方式生成单级单层路网
def generate_roadsNetwork_by_machanism(
        nodes_pos: np.ndarray = None,
        set_num_road_edges: int = None,
        set_num_interpolated_density_distance: float = None,
        num_road_edges_per_node: int = None,
        num_neighbors: int = None,
        roads_network_mechanism: str = 'Voronoi',
        network_type: str = 'Voronoi',
        is_preview_plot: bool = False,
        # level: int = 1,
        # max_levels: int = 10
):
    """
    生成路网地图，通过生成机制。

    这个方法的步骤是：
    1. 通过网络生成机制生成网络；
    4. 根据网络拓扑结构，生成实际的路径边；
    5. 对每条边插值生成路由节点；


    其中，网络类型有以下可选项：
    - 'Voronoi'：使用 Voronoi 图生成网络；
    - 'Complete'：使用完全图生成网络；
    - 'Random'：使用随机图生成网络；
    - 'Community Structure'：使用社区结构生成网络；
    - 'Small World'：使用小世界网络生成网络；
    - 'Scale Free'：使用无标度网络生成网络；
    - 'Hierarchical'：使用分层网络生成网络；
    - 'Regular'：使用规则网络生成网络；
    - 'Grid'：使用网格网络生成网络；
    - 'Scale Free'：使用无标度网络生成网络；




    Args:
        nodes_pos (np.ndarray): 节点空间坐标
        set_num_road_edges (int): 设置边数量
        set_num_interpolated_density_distance (float): 设定每条路径插值密度距离
        num_road_edges_per_node (int): 每个节点的边数量
        num_neighbors (int): 邻居节点数量
        roads_network_mechanism (str): 网络生成机制
        is_preview_plot (bool): 是否预览绘制。默认是 False。

    Returns:
        g (nx.DiGraph): 有向图网络

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
            - interpolated_points: 路由点（插值点）。数据结构是 np.ndarray，每一行是一个插值点的位置坐标，每一列是一个坐标维度
            - interpolated_points_path_length: 插值点沿着边方向到关键节点的路径长度向量。数据结构是 np.ndarray，每一个元素是该边之一个插值点到关键节点的路径长度




        第 2 个返回值根据不同的网络生成机制，返回不同的数据结构：
        - 'Voronoi'：返回 Voronoi 图相关的信息；

    """
    g, voronoi_graph = generate_network(nodes_pos=nodes_pos, graph_direction_type='directed', network_type='Voronoi', is_preview_plot=is_preview_plot)
    # roads_edges, voronoi_graph = _generate_weighted_voronoi_network(nodes_pos, is_preview_plot=True)

    # distance_matrix = _calculate_distance_matrix(nodes_pos,is_sparse=False)  # 计算节点之间的距离
    # neighbors = _select_neighbors(distance_matrix, len(nodes_pos), num_neighbors)  # 选择邻居节点
    # roads_edges = _create_road_edges(nodes_pos, neighbors, num_road_edges_per_node)  # 生成边
    ## 生成路由节点
    for edge in g.edges:
        node1 = nodes_pos[edge[0]]
        node2 = nodes_pos[edge[1]]
        interpolated_points_per_edge = _interpolate_points_by_linear_distance(node1, node2, set_num_interpolated_density_distance)
        g.edges[edge]['interpolated_points'] = interpolated_points_per_edge
        pass  # for

    # 计算主节点之间的距离
    for edge in g.edges:
        id_node1 = edge[0]
        id_node2 = edge[1]
        straight_distance = np.linalg.norm(nodes_pos[id_node1] - nodes_pos[id_node2])
        g.edges[edge]['straight_distance'] = straight_distance
        pass

    # 计算主节点之间的路径长度，计算每一个路由节点到邻近的主节点之间的路程长度
    for edge in g.edges:
        id_node1 = edge[0]
        id_node2 = edge[1]
        path_length = 0
        path_length += np.linalg.norm(nodes_pos[id_node1] - g.edges[edge]['interpolated_points'][0])
        interpolated_points_path_length = np.zeros(len(g.edges[edge]['interpolated_points']))
        for j in range(len(g.edges[edge]['interpolated_points']) - 1):
            path_length += np.linalg.norm(g.edges[edge]['interpolated_points'][j] - g.edges[edge]['interpolated_points'][j + 1])
            interpolated_points_path_length[j] = path_length
            pass  # for j
        path_length += np.linalg.norm(g.edges[edge]['interpolated_points'][-1] - nodes_pos[id_node2])
        g.edges[edge]['path_length'] = path_length
        g.edges[edge]['interpolated_points_path_length'] = interpolated_points_path_length[::-1]  # 插值点到关键节点路径长度
        pass  # for i, edge

    # if level < max_levels:
    #     for node in g.nodes:
    #         sub_nodes_pos = generate_points(
    #             set_densityDistance=set_city_nodes_density_distance,
    #             set_numPoints=None,
    #             set_circleRadius=set_city_circle_radius,
    #             circle_origin=nodes_pos[node],
    #             distribution='Posssion Disk'
    #         )
    #         sub_network, _ = generate_roadsNetwork_by_machanism(
    #             nodes_pos=sub_nodes_pos,
    #             set_num_interpolated_density_distance=1.0,
    #             is_preview_plot=False,
    #             level=level + 1,
    #             max_levels=max_levels
    #         )
    #         g.nodes[node]['sub_network'] = sub_network

    match network_type:
        case 'Voronoi':
            return g, voronoi_graph
        case _:
            errors = f"network_type {network_type} is not supported."
            raise ValueError(errors)
            pass  # match

    pass  # function


## #NOTE 根据导入图像生成路网


def generate_roadsNetwork_by_importImage(filepath_origin_image, filepath_signed_image):
    """
    根据导入原始图像及其标记图像，生成路网矢量信息。

    Args:
        filepath_origin_image (str): 原始图像路径
        filepath_signed_image (str): 标记图像路径

    Returns:
        roads_map (dict): 路网数据结构
        roads_skeleton_binary_map (np.ndarray): 道路骨架二值化像素图
        roads_skeleton_id_map (np.ndarray): 道路骨架像素图，像素点标记了所属边之 id。其中，-1 表示未标记道路 id 之道路像素，-2 表示不在道路骨架地图像素图之道路像素，-3 表示交叉点，-4 表示边之起点、终点像素，其它值表示道路 id。
    """
    # 导入图像

    # #NOTE 配置选项
    threshold_merge_near_junctions = 10  # 合并近邻交叉点的阈值
    min_length_of_pixels_of_edge = 10  # 路径边之最小像素长度
    interpolated_density_distance = 4  # 插值节点插值密度距离

    # 配置限制
    original_max_pixels = Image.MAX_IMAGE_PIXELS  # 保存原始配置，以便恢复
    Image.MAX_IMAGE_PIXELS = None  # 解除最大像素限制

    # 提取像素图像中的道路骨架，并保存为图像。
    # Load images
    image_origin = io.imread(filepath_origin_image)
    image_signed = io.imread(filepath_signed_image)
    image_RGB = image_signed[:, :, :3]

    # 提取红色标记像素
    area_red_signed_pixels = np.all(image_RGB == [255, 0, 0], axis=-1)
    roads_signed_pixels = area_red_signed_pixels

    # 膨胀以扩展道路宽度
    dilated = morphology.dilation(roads_signed_pixels, morphology.square(3))

    # 使用连接组件标记分割道路
    pixels_signed_roads_dilated = measure.label(dilated)

    # 骨架化每个连接组件
    skeletons = []
    for i in range(1, pixels_signed_roads_dilated.max() + 1):
        mask = pixels_signed_roads_dilated == i
        skeleton = morphology.skeletonize(mask)
        skeletons.append(skeleton)

    # 将骨架合并为单个图像
    img_image_skeleton = np.zeros_like(skeletons[0])
    for skeleton in skeletons:
        img_image_skeleton = np.logical_or(img_image_skeleton, skeleton)

    # # 可选，保存骨架图像
    # io.imsave(filepath_output_image, img_image_skeleton.astype(np.uint8) * 255, check_contrast=False)

    # img_image_skeleton = io.imread(str(filepath_input_image))

    # 转换为灰度图像
    # 判断是几通道图像
    if len(img_image_skeleton.shape) == 3:
        gray = color.rgb2gray(img_image_skeleton)
    else:  # 如果是单通道图像
        gray = img_image_skeleton
        pass  # if

    # 道路骨架地图像素图。使用大津阈值法做二值化。标记了道路像素的值为 True，否则为 False。
    roads_skeleton_binary_map = gray > filters.threshold_otsu(gray)

    # 创建一个空的路网数据结构
    roads_map = {
        'junctions_nodes': {},  # 交叉节点
        'road_edges': {},  # 路径边
        'interpolated_nodes': {},  # 插值节点
    }

    ## 生成交叉点集
    junction_nodes = _detect_junctions_points(roads_skeleton_binary_map, connectivity_limit=26)

    # 融合邻近的交叉点
    junction_nodes = _merge_near_junctions(junction_nodes, threshold_merge_near_junctions)

    ## 生成边集
    # 添加边集。边集数据结构为列表，每个元素为一个元组，元组有 3 个元素，分别是边 id 、边之两个交叉节点下标、边之像素点空间坐标列表；
    road_edges = []

    roads_skeleton_id_map = np.zeros_like(roads_skeleton_binary_map, dtype=int)  # 用一个像素图像数组标记 binary 之道路像素所属边之 id
    roads_skeleton_id_map[roads_skeleton_binary_map] = -1  # 未标记道路 id 之道路像素值为 -1
    roads_skeleton_id_map[~roads_skeleton_binary_map] = -2  # 不在道路骨架地图像素图之道路像素的值为 -2
    for i, node in enumerate(junction_nodes):
        roads_skeleton_id_map[node[0], node[1]] = -3  # 标记为交叉点的像素值为 -3
        pass  # for

    # 遍历每一个节点
    # 遍历节点周围 True 值的相邻像素点（这里规定相邻是指8邻域相邻，暨像素点周围8个像素点）的边值为边下标：
    # 1. 类似画图工具里的填充工具那样，沿着其中一个相邻像素点出发出发，标记沿途经过的每一个是路径的相邻像素点的边值为边下标，直到遇到一个新的节点像素点，或者遇到已经标记边id的像素点为止；
    # 2. 此时标记一个边完毕。这个边需要记录边的id、边的两个节点id、边的所有像素点集合（按照沿途经过的添加顺序排序）；
    # #BUG 缺少图像边界检查
    # #BUG 缺少处理环边

    neighbors = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if i != 0 or j != 0]  # 8邻域
    # 遍历每一个节点，标记边之起点、终点像素
    for i, node in enumerate(junction_nodes):
        for j, k in neighbors:  # 遍历交叉节点之相邻像素点
            if roads_skeleton_id_map[node[0] + j, node[1] + k] == -1:
                roads_skeleton_id_map[node[0] + j, node[1] + k] = -4  # 标记为边的之点、终点像素
                pass  # if
            pass  # for j, k
        pass  # for i, node
    # 遍历每一个节点，标记各边所有的像素
    idx_edge = 0
    for i, node in enumerate(junction_nodes):
        for j, k in neighbors:  # 遍历交叉节点之相邻像素点
            x, y = node[0] + j, node[1] + k  # 待判断的前沿的像素点空间坐标
            # 判断路径边之起点
            if roads_skeleton_id_map[x, y] == -4:
                road_edges_terminal_1 = i  # 获取边之第一个端点下标
                # 沿着边像素点，一路沿着相邻边像素点出发，填充标记边缘像素点的边值为边下标，直到遇到边之起点、终点像素点为止
                is_continue_fill_edge = True
                edge_pixels = []
                while is_continue_fill_edge:
                    # 获取邻居像素点的值
                    neighbors_pixels = [(x + m, y + n) for m, n in neighbors]
                    neighbors_values = [roads_skeleton_id_map[x + m, y + n] for m, n in neighbors]
                    # print("当前像素点：", (x, y), "当前像素点值：", roads_skeleton_id_map[x, y])  # DEBUG
                    # print("邻居像素点：", neighbors_pixels)  # DEBUG
                    # print("邻居像素点值：", neighbors_values)  # DEBUG

                    # 处理当前像素点
                    if roads_skeleton_id_map[x, y] == -1:  # 如果当前像素点值是 -1 的
                        roads_skeleton_id_map[x, y] = idx_edge  # 标记当前所在像素点为路径边 id
                        edge_pixels.append((x, y))  # 追加边之像素点
                        # 前进到接下来要填充的邻居像素点。优先选择 -4 的像素点，其次选择 -1 的像素点，这样能够避免误填充其它路径边之像素点
                        if -4 in neighbors_values:
                            x, y = neighbors_pixels[neighbors_values.index(-4)]
                        elif -1 in neighbors_values:
                            x, y = neighbors_pixels[neighbors_values.index(-1)]
                        elif max(neighbors_values) >= 0 and min(neighbors_values) == -2:  # 如果没有遇到上述类型的像素点，且遇到的邻居像素点出现别的路径边之像素点，或者周围的像素点，则说明这个路径边出现偏差，需要特殊处理
                            road_edges_terminal_2, _ = find_nearest_point((x, y), junction_nodes)  # 寻找距离最近的交叉点作为该路径边之第二个端点
                            is_continue_fill_edge = False
                            pass  # if
                    elif roads_skeleton_id_map[x, y] == -4:  # 如果当前像素点值是 -4 的
                        roads_skeleton_id_map[x, y] = idx_edge  # 标记当前所在像素点为路径边 id
                        edge_pixels.append((x, y))  # 追加边之像素点
                        # 判断邻居像素点
                        if -3 in neighbors_values:  # 如果邻居像素点包含交叉点
                            # 判断交叉点对于该路径边是什么类型的端点
                            if neighbors_pixels[neighbors_values.index(-3)] != (node[0], node[1]):  # 如果邻居像素点不是该路径边之第一个端点下标，则说明当前像素点是该路径边之起点 #BUG 如果邻居存在多个交叉点的极端情况，可能会出现恐慌
                                road_edges_terminal_2 = neighbors_values.index(-3)
                                edge_pixels.append((x, y))  # 追加边之像素点
                                is_continue_fill_edge = False  # 不再填充该路径
                                is_this_edge_valid = True  # 这个路径边合法
                            else:  # 如果邻居像素点是该路径边之第一个端点下标，则说明当前像素点是该路径边之终点
                                # 前进到接下来要填充的邻居像素点
                                if -1 in neighbors_values:
                                    x, y = neighbors_pixels[neighbors_values.index(-1)]
                                else:  # 周围都没有遇到 -1 的像素点，说明这个路径边出现偏差，需要特殊处理
                                    road_edges_terminal_2, _ = find_nearest_point((x, y), junction_nodes)  # 寻找距离最近的交叉点作为该路径边之第二个端点
                                    is_continue_fill_edge = False
                                    pass  # if
                                pass  # if 判断交叉点对于该路径边是什么类型的端点

                            pass  # if 判断邻居像素点
                        pass  # if 处理当前像素点
                    pass  # while

                # # #DEBUG 可视化 roads_skeleton_id_map
                # roads_skeleton_id_map_RGB = np.zeros((roads_skeleton_id_map.shape[0], roads_skeleton_id_map.shape[1], 3), dtype=np.uint8)
                # roads_skeleton_id_map_RGB[roads_skeleton_id_map == -2] = [0, 0, 0]  # 非路径边填充黑色
                # roads_skeleton_id_map_RGB[roads_skeleton_id_map == -4] = [255, 0, 0]  # 路径端点填充红色
                # roads_skeleton_id_map_RGB[roads_skeleton_id_map == -3] = [0, 0, 255]  # 交叉点填充蓝色
                # roads_skeleton_id_map_RGB[roads_skeleton_id_map == -1] = [127, 127, 127]  # 未填充过的路径边填充灰色
                # roads_skeleton_id_map_RGB[roads_skeleton_id_map >= 0] = [255, 255, 0]  # 已填充过的路径边填充黄色
                # roads_skeleton_id_map_RGB[roads_skeleton_id_map == idx_edge] = [0, 255, 0]  # 当前填充完毕的路径边填充绿色
                # io.imsave(f"./roads_skeleton_id_map_RGB/{idx_edge}.png", roads_skeleton_id_map_RGB, check_contrast=False)  # 保存图像
                # plt.imshow(roads_skeleton_id_map_RGB, aspect='equal')  # 显示图像（按照原图尺寸）
                # plt.show()

                # 判断路径边是否像素点过少
                if len(edge_pixels) < min_length_of_pixels_of_edge:  # 如果路径边之像素点数量过小，说明这个路径边过短，可以不予考虑
                    is_this_edge_valid = False

                # 处理合法的路径边
                if is_this_edge_valid:
                    # 追加该边信息至边集
                    road_edges.append((
                        idx_edge,
                        (road_edges_terminal_1, road_edges_terminal_2),
                        edge_pixels,
                    ))
                    idx_edge += 1  # 边 id 自增
                    pass  # if

                pass  # if 判断路径边之起点
            pass  # for j, k in neighbors
        pass  # for i, node

    ## 插值节点
    interpolated_nodes = []  # 插值节点。每个元素为一个元组，元组有 4 个元素，分别是插值节点 id 、插值节点所属的边 id 、插值节点相邻节点对、插值节点空间坐标
    for i, edge in enumerate(road_edges):
        edge_pixels = edge[2]  # 假设 edge_info 结构为 (edge_id, (node1_id, node2_id), edge_pixels)
        interpolated_points_per_edge = _interpolate_points_along_pixels_path(edge_pixels, interpolated_density_distance)
        for j, point in enumerate(interpolated_points_per_edge):
            if j == 0:  # 如果是第一个插值节点，那么相邻节点对之第一个节点是边之第一个节点
                interpolated_nodes.append((
                    j,  # 插值节点 id
                    edge[0],  # 插值节点所属的边 id
                    (edge[1][0], j + 1),  # 插值节点相邻节点对
                    point,  # 插值节点空间坐标
                ))
            elif j == len(interpolated_points_per_edge) - 1:  # 如果是最后一个插值节点，那么相邻节点对之第二个节点是边之第二个节点
                interpolated_nodes.append((
                    j,  # 插值节点 id
                    edge[0],  # 插值节点所属的边 id
                    (j - 1, edge[1][1]),  # 插值节点相邻节点对
                    point,  # 插值节点空间坐标
                ))
            else:  # 如果是中间的插值节点，那么相邻节点对之第一个节点是上一个插值节点，第二个节点是下一个插值节点
                interpolated_nodes.append((
                    j,  # 插值节点 id
                    edge[0],  # 插值节点所属的边 id
                    (j - 1, j + 1),  # 插值节点相邻节点对
                    point,  # 插值节点空间坐标
                ))
            pass  # for j, point
        pass  # for i, edge

    ## 添加路网数据结构
    for i, node in enumerate(junction_nodes):
        roads_map['junctions_nodes'][i] = node
    for i, edge in enumerate(road_edges):
        roads_map['road_edges'][i] = road_edges[i]
    for i, interpolated_node in enumerate(interpolated_nodes):
        roads_map['interpolated_nodes'][i] = interpolated_nodes[i]

    # 恢复配置
    Image.MAX_IMAGE_PIXELS = original_max_pixels

    return roads_map, roads_skeleton_binary_map, roads_skeleton_id_map
    pass  # function


# def _detect_junctions_points(skeleton, connectivity_limit=26):
#     """
#     检测骨架图像中的交叉点。
#
#     Args:
#         skeleton (np.ndarray): 骨架图像
#         connectivity_limit (int): 连通性限制最大值，默认为 26
#
#     Returns:
#         np.array: Array of junction points.
#     """
#
#     from skimage.morphology import skeletonize_3d, thin
#
#     # 对骨架图像进行3D细化
#     skeleton_3d = skeletonize_3d(np.expand_dims(skeleton, axis=2))
#
#     # 提取线段数
#     line_segments = thin(skeleton_3d)
#     line_segment_count = np.squeeze(line_segments, axis=2)
#
#     # 三条线交叉的点就是交叉点
#     junctions = line_segment_count == 3
#
#     return np.where(junctions)
#     pass  # function


def _detect_junctions_points(skeleton, connectivity_limit=26):
    """
    检测骨架图像中的交叉点。

    Args:
        skeleton (np.ndarray): 骨架图像
        connectivity_limit (int): 连通性限制最大值，默认为 26

    Returns:
        np.array: Array of junction points.
    """
    from scipy.ndimage import convolve

    # 定义用于检测连接的卷积核
    kernel = np.array(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]]
    )

    # 对骨架图像进行卷积，获取每个像素的邻居数量
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)

    # 初始化交叉点数组空间坐标
    junctions = []

    # 遍历骨架图像中的每个像素
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 1:  # 如果当前像素是骨架的一部分
                # 获取当前像素的8个邻居
                neighbors = skeleton[y - 1:y + 2, x - 1:x + 2]
                # 计算邻居数量
                count = neighbor_count[y, x]

                # # 如果邻居数量为3，则可能是交叉点 #HACK 这个已经无用
                # if count >= 3:
                #     # 检查这3个邻居是否非连续
                #     if _is_three_non_continuous_neighbors(neighbors):
                #         junctions.append((y, x))

                # 如果邻居数量为3，则可能是交叉点
                if count >= 3:
                    junctions.append((y, x))

                # 如果邻居数量为1，则可能是端点
                if count == 1:
                    junctions.append((y, x))

    return junctions
    pass  # function


# def _is_three_non_continuous_neighbors(neighbors):
#     """
#     检测3x3邻居数组中是否有3个非连续的邻居。
#
#     Args:
#         neighbors (np.ndarray): 3x3 数组，表示一个像素的8个邻居
#
#     Returns:
#         bool: 如果有3个非连续的邻居，则返回 True，否则返回 False。
#     """
#     center = (1, 1)  # 中心像素的位置
#     neighbor_positions = [(0, 0), (0, 1), (0, 2),
#                           (1, 0), (1, 2),
#                           (2, 0), (2, 1), (2, 2)]  # 8邻居位置
#
#     # 获取8邻居的值
#     neighbor_values = [(row, col, neighbors[row, col]) for row, col in neighbor_positions]
#
#     # 获取骨架像素的邻居
#     skeleton_neighbors = [(row, col) for row, col, value in neighbor_values if value == 1]
#
#     if len(skeleton_neighbors) != 3:
#         return False
#
#     # 检查3个邻居是否非连续，即是否有两个邻居的行或列空间坐标之差大于1
#     for i, (row1, col1) in enumerate(skeleton_neighbors):
#         for j, (row2, col2) in enumerate(skeleton_neighbors):
#             if i != j and abs(row1 - row2) <= 1 and abs(col1 - col2) <= 1:
#                 # 两个邻居的行或列空间坐标之差小于等于1，说明它们是连续的
#                 return False
#
#     # 如果3个邻居都非连续，则返回 True
#     return True
#
#     pass  # function


def _merge_near_junctions(junctions, threshold=3):
    """
    融合过于接近的交叉点。

    Args:
        junctions (list of tuples): 交叉点空间坐标列表，每个元素是一个包含两个元素的元组，表示交叉点的空间坐标。
        threshold (int): 融合阈值，单位为像素。

    Returns:
        list of tuples: 融合后的交叉点空间坐标列表。
    """
    if len(junctions) == 0:
        return []

    # 将交叉点列表转换为NumPy数组以便进行计算
    junctions_array = np.array(junctions)

    # 初始化一个空列表来存储融合后的交叉点
    merged_junctions = []

    # 使用一个数组标记哪些交叉点已经被处理过
    processed = np.zeros(len(junctions), dtype=bool)

    for i, junction in enumerate(junctions_array):
        if not processed[i]:
            # 找到与当前交叉点距离小于阈值的所有交叉点
            distances = np.sqrt(np.sum((junctions_array - junction) ** 2, axis=1))
            close_by = distances < threshold  # 与当前交叉点距离小于阈值的交叉点的索引

            # 计算这些交叉点的平均空间坐标，
            mean_junction = np.mean(junctions_array[close_by], axis=0)

            # #DEBUG 找到该平均空间坐标距离这些交叉点空间坐标当中最近的一个空间坐标，该空间坐标作为融合后的交叉点所在的空间坐标
            merged_junction = junctions_array[close_by][np.argmin(np.sum((junctions_array[close_by] - mean_junction) ** 2, axis=1))]

            # 将融合后的交叉点添加到列表中
            merged_junctions.append(tuple(merged_junction))

            # 标记已处理的交叉点
            processed[close_by] = True

    return merged_junctions
    pass  # function


def _process_road_images(filepath_origin_image, filepath_signed_image, filepath_output_image):
    """
    提取像素图像中的道路骨架，并保存为图像。

    Args:
        filepath_origin_image (str): 原始图像路径
        filepath_signed_image (str): 标记图像路径
        filepath_output_image (str): 输出图像路径

    Returns:
        final_skeleton (np.ndarray): 道路骨架图像
    """
    # Load images
    image_origin = io.imread(filepath_origin_image)
    image_signed = io.imread(filepath_signed_image)
    image_RGB = image_signed[:, :, :3]

    # Extract red-signed pixels
    area_red_signed_pixels = np.all(image_RGB == [255, 0, 0], axis=-1)
    roads_signed_pixels = area_red_signed_pixels

    # Dilate to expand road width
    dilated = morphology.dilation(roads_signed_pixels, morphology.square(3))

    # Segment roads using connected component labeling
    pixels_signed_roads_dilated = measure.label(dilated)

    # Skeletonize each connected component
    skeletons = []
    for i in range(1, pixels_signed_roads_dilated.max() + 1):
        mask = pixels_signed_roads_dilated == i
        skeleton = morphology.skeletonize(mask)
        skeletons.append(skeleton)

    # Combine skeletons into a single image
    final_skeleton = np.zeros_like(skeletons[0])
    for skeleton in skeletons:
        final_skeleton = np.logical_or(final_skeleton, skeleton)

    # Optionally, save the final skeleton image
    io.imsave(filepath_output_image, final_skeleton.astype(np.uint8) * 255)

    return final_skeleton
    pass  # function


def _calculate_cumulative_distances(pixels):
    """
    计算像素点之间的累积距离。

    Args:
        pixels (list of tuples): 像素点空间坐标列表

    Returns:
        list: 累积距离列表
    """
    distances = [0]
    for i in range(1, len(pixels)):
        distance = np.sqrt((pixels[i][0] - pixels[i - 1][0]) ** 2 + (pixels[i][1] - pixels[i - 1][1]) ** 2)
        distances.append(distances[-1] + distance)
    return distances
    pass  # function


def _interpolate_points_by_num_of_points(node1: np.ndarray, node2: np.ndarray, num_points):
    """
    根据每条边的插值点数生成插值点

    Args:
        node1 (np.ndarray): 起始节点空间坐标
        node2 (np.ndarray): 终止节点空间坐标
        num_points (int): 插值点数

    Returns:
        interpolated_points (np.ndarray): 插值点空间坐标
    """
    t = np.linspace(1, num_points, num_points) / (num_points + 1)
    interpolated_points = node1 + t[:, np.newaxis] * (node2 - node1)
    return interpolated_points


# def _interpolate_points_by_num_of_points(node1: jnp.array, node2: jnp.array, num_points):
#     """
#     根据每条边的插值点数生成插值点（JAX 版本）
#
#     Args:
#         node1 (jnp.array): 起始节点空间坐标
#         node2 (jnp.array): 终止节点空间坐标
#         num_points (int): 插值点数
#
#     Returns:
#         interpolated_points: list, 插值点空间坐标
#     """
#     t = jnp.linspace(1, num_points, num_points) / (num_points + 1)
#     interpolated_points = node1 + t[:, jnp.newaxis] * (node2 - node1)
#     return interpolated_points.tolist()


def _interpolate_points_by_linear_distance(node1: np.ndarray, node2: np.ndarray, interpolate_distance):
    """
    根据每条边的插值距离生成线性插值点空间坐标列表

    Args:
        node1 (np.ndarray): 起始节点空间坐标
        node2 (np.ndarray): 终止节点空间坐标
        interpolate_distance (float): 插值距离

    Returns:
        interpolated_points (np.ndarray): 插值点空间坐标
    """
    length = np.linalg.norm(node2 - node1)
    num_points = int(length / interpolate_distance)
    t = np.linspace(0, 1, num_points + 2)[1:-1]
    interpolated_points = node1 + t[:, np.newaxis] * (node2 - node1)
    return interpolated_points


# @jit
# def _interpolate_points_by_linear_distance(node1, node2, interpolate_distance):
#     """
#     根据每条边的插值距离生成线性插值点空间坐标列表（JAX版本）
#
#     Args:
#         node1 (jnp.array): 起始节点空间坐标
#         node2 (jnp.array): 终止节点空间坐标
#         interpolate_distance (float): 插值距离
#
#     Returns:
#         interpolated_points: list, 插值点空间坐标
#     """
#     length = jnp.linalg.norm(node2 - node1)
#     num_points = int(length / interpolate_distance)
#     t = jnp.linspace(0, 1, num_points + 2)[1:-1]
#     interpolated_points = node1 + t[:, jnp.newaxis] * (node2 - node1)
#     return interpolated_points.tolist()


def _interpolate_points_along_pixels_path(pixels, distance):
    """
    沿着给定的像素路径生成插值点空间坐标列表。

    Args:
        pixels (list of tuples): 像素点空间坐标列表
        distance (float): 插值距离

    Returns:
        list: 插值点空间坐标列表
    """

    if not pixels or len(pixels) < 2:
        return []

    cumulative_distances = _calculate_cumulative_distances(pixels)
    total_distance = cumulative_distances[-1]
    remainder = total_distance % distance
    num_points = int(total_distance / distance)
    adjusted_distance = distance + remainder / num_points  # 重新微调插值距离

    interpolated_points = []
    for i in range(1, num_points):
        target_distance = i * adjusted_distance
        for j in range(1, len(cumulative_distances)):
            if cumulative_distances[j] >= target_distance:
                ratio = ((target_distance - cumulative_distances[j - 1]) /
                         (cumulative_distances[j] - cumulative_distances[j - 1]))
                x = pixels[j - 1][0] + ratio * (pixels[j][0] - pixels[j - 1][0])
                y = pixels[j - 1][1] + ratio * (pixels[j][1] - pixels[j - 1][1])
                interpolated_points.append((x, y))
                break
    return interpolated_points
    pass  # function
