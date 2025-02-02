"""
导入简易的分级分层的道路交通网络地图，预计算各级网络各点之间的最短路径，包括最短距离和最短路径。

设定：
- 这里的交通网络是二级交通网络，顶级网络是全国交通网络，二级网络是城市交通网络。
- 各级网络节点包括两种：一种是关键节点，一种是路由节点。
- 二级关键节点包括两种，一种是非城市出入口的关键节点，一种是城市出入口的关键节点。

多级网络结构为：邻居级别的网络之出入口节点都是一个的情况，意思是二级网络的每一个城市出入口节点只有一个。类似于每一个城市一个火车站（不考虑多个火车站的情况）。
多级网络结构为：邻居级别的网络之出入口节点分属不同节点的情况，意思是二级网络的每一个城市出入口节点有多个。类似于每一个城市有多个高速公路出入口站。每一个高速公路出入口站一一对应该城市的一座邻居城市。

图数据结构如下：

g = nx.DiGraph()
    g.nodes:
        - id (int): 节点 id
        - name (str): 节点名称。格式是 `f"城市={city:0{m}d}"`
        - type (str): 节点类型。包括：'城市节点'
        - pos (np): ndarray。节点位置坐标
        - sub_hierarchy_network (nx.Graph): 城市内部交通网络
        - city_io_nodes (dict): 城市出入口节点字典，键是邻居节点 id，值是城市内部节点 id
    g.edges:
        - id (int): 边 id
        - name (str): 边名称。格式是 `f"公路={edge[0]:0{m}d}-{edge[1]:0{m}d}"`
        - type (str): 边类型。包括：'公路'
        - straight_distance (float): 直线距离
        - path_length (float): 路径长度
        - interpolated_points: 路由点（插值点）。数据结构是 np.ndarray，每一行是一个插值点的位置坐标，每一列是一个坐标维度
        - interpolated_points_path_length: 插值点沿着边方向到关键节点的路径长度向量。数据结构是 np.ndarray，每一个元素是该边之一个插值点到关键节点的路径长度
    g.graph:
        - name (str): 名称。内容是 '全国交通网络'
        - type (str): 类型。内容是 '有向无环分级图'
        - network_shortest_paths (dict): 全国交通网络各级网络各点之间的最短路径嵌套字典。键是起点 id，值是终点 id 和最短路径列表。最短路径列表存储的是节点 id 构成的路径。
        - network_shortest_paths_length (dict): 全国交通网络各级网络各点之间的最短路程嵌套字典。键是起点 id，值是终点 id 和最短路程。
        - description (str): 描述。
        - notes (str): 备注。

其中，城市内部交通网络数据格式如下：
    g.nodes:
        - id (int): 节点 id
        - name (str): 节点名称。格式是 `f"节点={inter_city_node:0{m}d}"`、`f"城市出入口={city_io_nodes[id_neighbor]:0{m}d}-{id_neighbor:0{m}d}"`
        - type (str): 节点类型。包括：'城内交通节点'、'城市出入口'
        - pos (np): ndarray。节点位置坐标
        - city_io_nodes (dict): 城市出入口节点字典，键是邻居节点 id，值是城市内部节点 id
    g.edges:
        - id (int): 边 id
        - name (str): 边名称。格式是 `f"街道={edge[0]:0{m}d}-{edge[1]:0{m}d}"`
        - type (str): 边类型。包括：'街道'
        - straight_distance (float): 直线距离
        - path_length (float): 路径长度
        - interpolated_points: 路由点（插值点）。数据结构是 np.ndarray，每一行是一个插值点的位置坐标，每一列是一个坐标维度
        - interpolated_points_path_length: 插值点沿着边方向到关键节点的路径长度向量。数据结构是 np.ndarray，每一个元素是该边之一个插值点到关键节点的路径长度
    g.graph:
        - city_io_nodes (dict): 城市出入口节点字典，键是邻居城市节点 id，值是该邻居城市连接的城市出入口节点 id

"""

# %% 导入库
import pickle
from pathlib import Path
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from complex_network_operation.utils.tools import Tools

# %% 导入生成的简易的分级分层的道路交通网络地图

folderpath_project = Path(Tools.get_project_rootpath('ComplexNetworkOperation', '.')).resolve()
folderpath_input_data = folderpath_project / 'data/分级道路交通网络地图'

with open(Path(folderpath_input_data / 'city_traffic_networks.pkl'), 'rb') as f:
    g_world_city_traffic_network = pickle.load(f)

# %% # 预计算各个同级网络之各关键节点之间的最短路径、最短路程
g_world_city_traffic_network_shortest_paths = dict()  # 全国交通网络各级网络各点之间的最短路径
g_world_city_traffic_network_shortest_paths_length = dict()  # 全国交通网络各级网络各点之间的最短距离
g_world_city_traffic_network.graph['network_shortest_paths'] = dict(nx.all_pairs_dijkstra_path(g_world_city_traffic_network, weight='path_length'))  # 世界各城市交通网络各级网络各点之间的最短路径
g_world_city_traffic_network.graph['network_shortest_paths_length'] = dict(nx.all_pairs_dijkstra_path_length(g_world_city_traffic_network, weight='path_length'))  # 世界各城市交通网络各级网络各点之间的最短距离
# 各城市之城区道路交通网络节点之间的最短路径、最短路程
for id_city_node in g_world_city_traffic_network.nodes:
    g_city_traffic_network = g_world_city_traffic_network.nodes[id_city_node]['sub_hierarchy_network']
    g_city_traffic_network_shortest_paths = dict(nx.all_pairs_dijkstra_path(g_city_traffic_network, weight='path_length'))  # 城市内之城市道路交通网络节点之间的最短路径
    g_city_traffic_network_shortest_paths_length = dict(nx.all_pairs_dijkstra_path_length(g_city_traffic_network, weight='path_length'))  # 城市内之城市道路交通网络节点之间的最短距离
    g_world_city_traffic_network.nodes[id_city_node]['sub_hierarchy_network'].graph['network_shortest_paths'] = g_city_traffic_network_shortest_paths  # 各城市之城区道路交通网络之最短路径
    g_world_city_traffic_network.nodes[id_city_node]['sub_hierarchy_network'].graph['network_shortest_paths_length'] = g_city_traffic_network_shortest_paths_length  # 各城市之城区道路交通网络之最短距离
    pass  # for

# %% 预计算多级网络各点之间的最短路径、最短路程
# %% #NOTE NOW 当多级网络不考虑起点终点在路由节点的情况，且不考虑不同的出入口节点路径影响总的最短路径，预计算多级网络各点之间的最短路径、最短路程
world_city_node_start = 2
city_node_start = 0
world_city_node_end = 5
city_node_end = 1

node_start = [world_city_node_start, city_node_start]
node_end = [world_city_node_end, city_node_end]

# 世界交通网络最短路径、最短路程
shortest_path_in_world = g_world_city_traffic_network.graph['network_shortest_paths'][world_city_node_start][world_city_node_end]
shortest_path_length_in_world = g_world_city_traffic_network.graph['network_shortest_paths_length'][world_city_node_start][world_city_node_end]
# 起点的城市交通网络最短路径、最短路程
g_city_traffic_network_start = g_world_city_traffic_network.nodes[world_city_node_start]['sub_hierarchy_network']
city_io_node_start = g_city_traffic_network_start.graph['city_io_nodes'][shortest_path_in_world[1]]
shortest_path_in_city_start = g_city_traffic_network_start.graph['network_shortest_paths'][world_city_node_start][city_io_node_start]
shortest_path_length_in_city_start = g_city_traffic_network_start.graph['network_shortest_paths_length'][world_city_node_start][city_io_node_start]
# 终点的城市交通网络最短路径、最短路程
g_city_traffic_network_end = g_world_city_traffic_network.nodes[world_city_node_end]['sub_hierarchy_network']
city_io_node_end = g_city_traffic_network_end.graph['city_io_nodes'][shortest_path_in_world[-2]]
shortest_path_in_city_end = g_city_traffic_network_end.graph['network_shortest_paths'][city_io_node_end][world_city_node_end]
shortest_path_length_in_city_end = g_city_traffic_network_end.graph['network_shortest_paths_length'][city_io_node_end][world_city_node_end]
# 合并最短路径、最短路程
shortest_path = shortest_path_in_city_start + shortest_path_in_world + shortest_path_in_city_end
shortest_path_level = np.array([2] * len(shortest_path_in_city_start) + [1] * len(shortest_path_in_world) + [2] * len(shortest_path_in_city_end))
shortest_path_length = shortest_path_length_in_city_start + shortest_path_length_in_world + shortest_path_length_in_city_end

# %% #NOTE TODO 当多级网络考虑起点终点在路由节点的情况，且不考虑不同的出入口节点路径影响总的最短路径，预计算多级网络各点之间的最短路径、最短路程


# %% #NOTE TODO 当多级网络不考虑起点终点在路由节点的情况，且考虑不同的出入口节点路径影响总的最短路径，预计算多级网络各点之间的最短路径、最短路程
# 计算从一个城市的某一个城内关键节点到另一个城市的城内关键节点之间的最短路径、最短距离。
# 寻路算法举例：从城市A的某个城内节点（起点）到城市B的某个城内节点（终点）。期间需要先从顶级寻路城市A到城市B的最优路径。然后分别加上从城市A的起点到出口节点的最优路径、从城市B的入口节点到终点的最优路径。

# world_route_node_start = 5
world_city_node_start = 2
city_node_start = 0
# city_route_node_start = 0
# world_route_node_end = 10
world_city_node_end = 5
city_node_end = 1
# city_route_node_end = 1


# 在世界交通网络级别，计算起点到终点所在节点之间的最短路径
if world_city_node_start == world_city_node_end:  # 判断起点、终点是否是同一个节点，如果是则不计算
    shortest_path_in_net = [city_node_start, city_node_end]
    shortest_path_length_in_net = 0
else:
    # 获取世界交通网络起点、终点之邻居节点
    neighbors_start = g_world_city_traffic_network.neighbors(world_city_node_start)
    neighbors_end = g_world_city_traffic_network.neighbors(world_city_node_end)
    if world_city_node_end in neighbors_start:  # 判断起点、终点是否是邻居节点
        shortest_path_in_net = [city_node_start, city_node_end]
        shortest_path_length_in_net = g_world_city_traffic_network[world_city_node_start][world_city_node_end]['path_length']
    elif world_city_node_start in neighbors_end:
        shortest_path_in_net = [city_node_start, city_node_end]
        shortest_path_length_in_net = g_world_city_traffic_network[world_city_node_start][world_city_node_end]['path_length']
    else:
        # 获取起点各邻居节点到终点各邻居节点的路径列表、路程列表
        # 计算在同级网络中，起点节点经由起点邻居节点到终点邻居节点到终点节点的最短路径、最短距离
        dict_shortest_paths_in_net = dict()
        dict_shortest_paths_length_in_net = dict()
        for neighbor_start in neighbors_start:
            for neighbor_end in neighbors_end:
                shortest_path_in_net = g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths'][world_city_node_start][neighbor_start] + g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths'][neighbor_start][neighbor_end] + g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths'][neighbor_end][world_city_node_end]
                shortest_path_length_in_net = g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths_length'][world_city_node_start][neighbor_start] + g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths_length'][neighbor_start][neighbor_end] + g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths_length'][neighbor_end][world_city_node_end]
                dict_shortest_paths_in_net[(neighbor_start, neighbor_end)] = shortest_path_in_net
                dict_shortest_paths_length_in_net[(neighbor_start, neighbor_end)] = shortest_path_length_in_net
                pass  # for
            pass  # for
        pass  # if

# 在城市交通网络级别，计算起点到终点所在节点之间的最短路径


# 获取世界交通网络起点、终点之邻居节点
neighbors_start = g_world_city_traffic_network.neighbors(world_city_node_start)
neighbors_end = g_world_city_traffic_network.neighbors(world_city_node_end)
# 判断起点、终点是否是邻居节点


top_level_path = g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths'][world_city_node_start][world_city_node_end]
top_level_distance = g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths_length'][world_city_node_start][world_city_node_end]

# 获取起点城市内的最优路径
start_city_network = g_world_city_traffic_network.nodes[world_city_node_start]['city_traffic_network']
start_city_exit_node = top_level_path[1]  # 起点城市的出口节点
start_city_path = start_city_network[city_node_start][start_city_exit_node]
start_city_distance = start_city_network[city_node_start][start_city_exit_node]

# 获取终点城市内的最优路径
end_city_network = g_world_city_traffic_network.nodes[world_city_node_end]['city_traffic_network']
end_city_entry_node = top_level_path[-2]  # 终点城市的入口节点
end_city_path = end_city_network[end_city_entry_node][city_node_end]
end_city_distance = end_city_network[end_city_entry_node][city_node_end]

# 合并路径和距离
shortest_path = start_city_path + top_level_path[1:-1] + end_city_path
shortest_path_length = start_city_distance + top_level_distance + end_city_distance

# return shortest_path, shortest_path_length
# pass  # function

# #%%
# def calculate_intercity_shortest_path(world_city_node_start, city_node_start, world_city_node_end, city_node_end):
#     """
#     计算从一个城市的某一个城内节点到另一个城市的城内节点之间的最短路径、最短距离。
#
#     Args:
#         world_city_node_start (int): 起点城市ID
#         city_node_start (int): 起点城市内节点ID
#         world_city_node_end (int): 终点城市ID
#         city_node_end (int): 终点城市内节点ID
#
#     Returns:
#         shortest_path (list): 最短路径
#         shortest_path_length (float): 最短距离
#     """
#     # 获取顶级城市之间的最优路径
#     top_level_path = g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths'][world_city_node_start][world_city_node_end]
#     top_level_distance = g_world_city_traffic_network.graph['world_city_traffic_network_shortest_paths_length'][world_city_node_start][world_city_node_end]
#
#     # 获取起点城市内的最优路径
#     start_city_network = g_world_city_traffic_network.nodes[world_city_node_start]['city_traffic_network']
#     start_city_exit_node = top_level_path[1]  # 起点城市的出口节点
#     start_city_path = start_city_network[city_node_start][start_city_exit_node]
#     start_city_distance = start_city_network[city_node_start][start_city_exit_node]
#
#     # 获取终点城市内的最优路径
#     end_city_network = g_world_city_traffic_network.nodes[world_city_node_end]['city_traffic_network']
#     end_city_entry_node = top_level_path[-2]  # 终点城市的入口节点
#     end_city_path = end_city_network[end_city_entry_node][city_node_end]
#     end_city_distance = end_city_network[end_city_entry_node][city_node_end]
#
#     # 合并路径和距离
#     shortest_path = start_city_path + top_level_path[1:-1] + end_city_path
#     shortest_path_length = start_city_distance + top_level_distance + end_city_distance
#
#     return shortest_path, shortest_path_length
#     pass  # function


# %% #NOTE TODO 当多级网络考虑起点终点在路由节点的情况，且考虑不同的出入口节点路径影响总的最短路径，预计算多级网络各点之间的最短路径、最短路程
# 计算从一个城市的某一个城内路由节点到另一个城市的城内路由节点之间的最短路径、最短距离。
# 寻路算法举例：从城市A的某个城内节点（起点）到城市B的某个城内节点（终点）。期间需要先从顶级寻路城市A到城市B的最优路径。然后分别加上从城市A的起点到出口节点的最优路径、从城市B的入口节点到终点的最优路径。

world_route_node_start = 5
world_city_node_start = 2
city_node_start = 0
# city_route_node_start = 0
world_route_node_end = 10
world_city_node_end = 3
city_node_end = 1
# city_route_node_end = 1


# %% # 分页绘制全国交通网络及其各城市交通网络
# 绘制全国交通网络
world_city_pos = dict([(id_city_node, g_world_city_traffic_network.nodes[id_city_node]['pos']) for id_city_node in g_world_city_traffic_network.nodes])  # 标记节点位置
options_world_city_traffic_network = {
    'node_color': 'white',  # 节点颜色
    'linewidths': 2,  # 节点边框宽度
    'edgecolors': 'black',  # 节点边框颜色
    'node_size': 300,  # 节点大小
    'width': 2,  # 边宽度
    'with_labels': True,  # 是否显示节点标签
    'font_size': 12,  # 节点标签字体大小
    'font_color': 'black',  # 节点标签字体颜色
}
nx.draw_networkx(g_world_city_traffic_network, pos=world_city_pos, **options_world_city_traffic_network)

# %% # 可视化寻找路径

# 使用 NetworkX 可视化起点到终点的路径
# 绘制 g_city_traffic_network，标记节点 id、边路径长度
world_city_pos = dict([(id_city_node, g_world_city_traffic_network.nodes[id_city_node]['pos']) for id_city_node in g_world_city_traffic_network.nodes])

options_world_city_traffic_network = {
    'node_color': 'white',  # 节点颜色
    'linewidths': 2,  # 节点边框宽度
    'edgecolors': 'black',  # 节点边框颜色
    'node_size': 300,  # 节点大小
    'width': 2,  # 边宽度
    'with_labels': True,  # 是否显示节点标签
    'font_size': 12,  # 节点标签字体大小
    'font_color': 'black',  # 节点标签字体颜色
}

# 标记节点位置
pos = {id_city_node: g_world_city_traffic_network.nodes[id_city_node]['pos'] for id_city_node in g_world_city_traffic_network.nodes}
# 标记起点为绿色，终点为红色，其余是白色
options_world_city_traffic_network['node_color'] = ['green' if id_city_node == world_city_node_start else 'red' if id_city_node == world_city_node_end else 'white' for id_city_node in g_world_city_traffic_network.nodes]

# 标记路径边
print(shortest_path)
shortest_path_for_vis = list(zip(shortest_path[:-1], shortest_path[1:]))

# 标记路径颜色为黄色
edge_colors = [
    'yellow' if edge in zip(shortest_path[:-1], shortest_path[1:]) else 'black' for edge in g_world_city_traffic_network.edges()
]

nx.draw_networkx(g_world_city_traffic_network, pos=world_city_pos, **options_world_city_traffic_network)

# 绘制图像，包含3个子图，分别是：起点城市交通网络、全国交通网络、终点城市交通网络
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
# 绘制起点城市交通网络
g_city_traffic_network_start = g_world_city_traffic_network.nodes[world_city_node_start]['sub_hierarchy_network']
g_city_traffic_network_start_pos = dict([(id_city_node, g_city_traffic_network_start.nodes[id_city_node]['pos']) for id_city_node in g_city_traffic_network_start.nodes])
nx.draw_networkx(g_city_traffic_network_start, pos=g_city_traffic_network_start_pos, ax=axs[0], **options_world_city_traffic_network)
axs[0].set_title('start city traffic network')

# 绘制全国交通网络
nx.draw_networkx(g_world_city_traffic_network, pos=world_city_pos, ax=axs[1], **options_world_city_traffic_network)
nx.draw_networkx_edges(g_world_city_traffic_network, pos=world_city_pos, edge_color=edge_colors, ax=axs[1])
axs[1].set_title('world city traffic network')

# 绘制终点城市交通网络
g_city_traffic_network_end = g_world_city_traffic_network.nodes[world_city_node_end]['sub_hierarchy_network']
g_city_traffic_network_end_pos = dict([(id_city_node, g_city_traffic_network_end.nodes[id_city_node]['pos']) for id_city_node in g_city_traffic_network_end.nodes])
nx.draw_networkx(g_city_traffic_network_end, pos=g_city_traffic_network_end_pos, ax=axs[2], **options_world_city_traffic_network)
axs[2].set_title('end city traffic network')

# 绘制路径
nx.draw_networkx_edges(g_world_city_traffic_network, pos=world_city_pos, edgelist=shortest_path_for_vis, edge_color='yellow', ax=axs[1])

plt.show()
