# %% md
# 生成简易的分级分层的道路交通网络地图

"""
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

# %%
import logging, pickle
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from complex_network_operation.utils.tools import Tools

logging.basicConfig(level=logging.INFO)
# %%

# %%
# 参数设置
# N_cities = 16
# N_city_nodes = 16
set_world_circle_radius = 500
set_world_circle_density_distance = 250
set_world_city_num_interpolated_density_distance = 10.0
set_city_circle_radius = 50
set_city_nodes_density_distance = 25
set_num_interpolated_density_distance = 1.0

# %% md
# 生成各城市节点位置和城市间交通网络

# %%
from complex_network_operation.functions.generations.fun_generate_points import generate_points
from complex_network_operation.functions.generations.fun_generate_roads_map import generate_roadsNetwork_by_machanism

# %%
# 生成各城市节点位置
city_nodes_pos = generate_points(
    set_densityDistance=set_world_circle_density_distance,
    set_numPoints=None,
    set_circleRadius=set_world_circle_radius,
    circle_origin=np.array([0, 0]),
    distribution='Posssion Disk'
)

# 生成城市间交通顶级网络
g_world_city_traffic_network, city_networks_interpolated_points = generate_roadsNetwork_by_machanism(
    nodes_pos=city_nodes_pos,
    set_num_interpolated_density_distance=set_world_city_num_interpolated_density_distance,
    is_preview_plot=True
)

# %%
# 添加世界属性
g_world_city_traffic_network.graph['name'] = '全国交通网络'
g_world_city_traffic_network.graph['type'] = '有向无环分级图'
g_world_city_traffic_network.graph['description'] = \
    """
    有一种全国交通层次网络结构。假设在使用的时期内是静态的（不增删改城市、不增删改路），是双向的，是无环的，权重是正的，表示路径长度。全国的各个城市视为节点，通过覆盖全国的交通网络连接，这是个顶级网络。每一个城内部又是一个个的城内交通网络，暨城市的每一个道路交叉点视为一个节点，形成网络，这是低级网络。
    
    设定：
    
    - 假设不考虑火车、飞机，只考虑高速公路网络。
    - 设定对于低级网络，暨城内部交通网络节点集合，包括两类，一类是城内部节点集合，一类是沟通其他城市的高速出入口节点集合。顶级网络当中，该城市节点的的邻居，对应这该城市有相同个数的高速出入口节点。这些高速出入口节点分别与顶级网络节点连边对应
    - 假设顶级网络（高速公路网络）的通行效率远远高于城内部的通行效率。
    - 假设城市在顶级网络视为一个高速公路节点。不需要进入城内部的道路网络。路过中间城市的时候，而仅仅直接走顶级网络G在该城市的高速节点$G_j$，暨绕城高速节点，而不需要出入该城市j的高速出入口节点进入城内部。
    
    对于城内部的网络来说，按照是否路由节点划分，有两种类型：第一种是关键节点，也就是计算用的节点。另一种是路由节点，只是为了在关键节点之间做曲线连边插值的。关键节点按照内外部划分，有两种类型的节点：内部节点和外部节点。内部节点是不和其他城市网络连接的节点。外部节点就是高速公路出入口节点，与上一层次的网络的城市节点$G_i$对接。通过该类节点对应到$G_i$，从而连接到全国路网。一个城市的外部节点数量等于上一层次的网络的城市节点$G_i$的邻居节点数量。一个城市的外部节点与上一层次的网络的城市节点$G_i$的邻居节点一一对应。从交通网络角度来理解，就是每一个城市的每一个高速公路出入口一一对应其相邻城市。
    
    每一个城市 i 内部有自己的城市道路网络 $C_i$（低级）。全国高速是一个顶级网络 G，每一个节点是该城市的高速节点。全国高速在路过其他城市的时候，可以不进入城市，而是仅仅走高速节点。只有在出入城市的时候才涉及到城市道路网络 $C_i$。其中出入城市的道路网络与高速节点的连接是通过高速路口节点连接的。$C_i$的出入口节点就是通过G连接城市的邻居城市的节点集合。
    
    假设从$i=城市A$到$i=城市B$，第一步要经过$城市C$，最后一步是从$城市E$到$城市B$，那么规划的时候，$A$的出口节点只能是$A$到$C$的连接点，$B$的入口节点只能是$E$到$B$的连接点。也就是说，先在网络G规划A到B的路径$p1$。
    
    举例，假设$p1$的第一个节点是A，第二个节点是C，倒数第二个节点是E，最后一个节点是B。然后根据规划后的路径，再规划A内部的起点节点a1到高速出口节点$a_out_C$的路径$p2$，以及规划节点E和节点B的连接的高速入口节点$b_in_E$到城市B内部终点$b3$节点的路径。
    
    出入口节点其实在网络中仅仅根据路径的进出才区分了。好比说我要去东边的城市，就只能走东边的高速出口，而不能走西边的高速出口，因为从西边的高速出口走，就会走到别的城市，这样就走错路了。对于城市来说，离开的城市的高速出入口节点只和接下来一步要路过的邻居城市的顶级网络G对应的节点连接。进入的城市的高速出入口节点只和上一步路过的邻居城市的顶级网络G对应的节点连接。
    """

# %%
# 添加城市节点属性
m = len(str(g_world_city_traffic_network.number_of_nodes()))  # 城市数量对应的位数
for city in g_world_city_traffic_network.nodes:
    # 添加城市名称
    g_world_city_traffic_network.nodes[city]['name'] = f"城市={city:0{m}d}"
    # 添加城市类型
    g_world_city_traffic_network.nodes[city]['type'] = '城市节点'
    pass  # end for

# 添加城市间交通网络边属性
m = len(str(g_world_city_traffic_network.number_of_edges()))  # 城市间交通网络边数量对应的位数
for edge in g_world_city_traffic_network.edges:
    # 添加边名称
    g_world_city_traffic_network.edges[edge]['name'] = f"公路={edge[0]:0{m}d}-{edge[1]:0{m}d}"
    # 添加边类型
    g_world_city_traffic_network.edges[edge]['type'] = '公路'
    pass  # end for

# %%

# 对于每一个城市，依次生成城内交通网络
for id_city_node in g_world_city_traffic_network.nodes:

    # %% 生成城内交通网络节点位置
    city_nodes_pos = generate_points(
        set_densityDistance=set_city_nodes_density_distance,
        set_numPoints=None,
        set_circleRadius=set_city_circle_radius,
        circle_origin=g_world_city_traffic_network.nodes[id_city_node]['pos'],
        distribution='Posssion Disk',
    )

    # %% 生成城内交通网络
    g_city_traffic_network, city_networks_interpolated_points = generate_roadsNetwork_by_machanism(
        city_nodes_pos,
        set_num_interpolated_density_distance=set_num_interpolated_density_distance,
        is_preview_plot=False
    )

    # 添加城内交通节点属性
    m = len(str(g_city_traffic_network.number_of_nodes()))  # 城内交通网络节点数量对应的位数
    for inter_city_node in g_city_traffic_network.nodes:
        # 添加节点名称
        g_city_traffic_network.nodes[inter_city_node]['name'] = f"节点={inter_city_node:0{m}d}"
        # 添加节点类型
        g_city_traffic_network.nodes[inter_city_node]['type'] = '城内交通节点'
        pass

    # 添加城内交通网络边属性
    m = len(str(g_city_traffic_network.number_of_edges()))  # 城内交通网络边数量对应的位数
    for edge in g_city_traffic_network.edges:
        # 添加边名称
        g_city_traffic_network.edges[edge]['name'] = f"街道={edge[0]:0{m}d}-{edge[1]:0{m}d}"
        # 添加边类型
        g_city_traffic_network.edges[edge]['type'] = '街道'
        pass

    # %% 选取城内交通网络的节点作为城市的出入口节点
    # 计算城内交通网络的节点与邻居节点的距离，选取最近的那个节点作为与邻居节点对应的出入口节点
    ids_neighbors = list(g_world_city_traffic_network.neighbors(id_city_node))  # 获取该城市的邻居节点
    city_io_nodes = dict()  # 该城市的出入口节点。键是邻居节点 id，值是城市内部节点 id
    for id_neighbor in ids_neighbors:
        min_distance = np.inf
        city_io_node = None
        for i, node in enumerate(g_city_traffic_network.nodes):
            distance = np.linalg.norm(g_city_traffic_network.nodes[node]['pos'] - g_world_city_traffic_network.nodes[id_neighbor]['pos'])  # 计算城内交通网络的节点与邻居节点的距离
            if distance < min_distance:
                min_distance = distance
                city_io_node = node
                pass  # if
            pass  # for
        city_io_nodes[id_neighbor] = city_io_node  # 选取直线距离最近的那个节点作为与邻居节点对应的出入口节点
        g_city_traffic_network.nodes[city_io_nodes[id_neighbor]]['type'] = '城市出入口'
        g_city_traffic_network.nodes[city_io_nodes[id_neighbor]]['name'] = f"城市出入口={city_io_nodes[id_neighbor]:0{m}d}-{id_neighbor:0{m}d}"
        pass  # for

    # 添加该城市属性：出入口节点字典
    g_city_traffic_network.graph['city_io_nodes'] = city_io_nodes

    # %%  添加该城市交通网络到全国交通网络
    g_world_city_traffic_network.nodes[id_city_node]['sub_hierarchy_network'] = g_city_traffic_network

    pass  # for

# %%
# 保存生成的城市交通网络
folderpath_project = Path(Tools.get_project_rootpath('ComplexNetworkOperation', '.')).resolve()
folderpath_output_data = folderpath_project / 'data/分级道路交通网络地图'
# 保存 pickle 文件
with open(Path(folderpath_output_data, 'city_traffic_networks.pkl'), 'wb') as f:
    pickle.dump(g_world_city_traffic_network, f)
    # # 保存为 GraphML 文件
    # nx.write_graphml(g_world_city_traffic_network, "city_traffic_networks.graphml")
    # # 保存为 JSON 文件，以便于 Rust 或者 D3.js 等其他语言读取
    # nx.write_adjlist(g_world_city_traffic_network, "city_traffic_networks.adjlist")

# 备份 pickle 文件，后缀名加上时间戳
with open(Path(folderpath_output_data, f'city_traffic_networks_{pd.Timestamp.now().strftime("%Y%m%d%H%M%S")}.pkl'), 'wb') as f:
    pickle.dump(g_world_city_traffic_network, f)
    pass  # with

# %%
# 读取生成的城市交通网络
folderpath_project = Path(Tools.get_project_rootpath('ComplexNetworkOperation', '.')).resolve()
folderpath_input_data = folderpath_project / 'data/分级道路交通网络地图'
with open(Path(folderpath_input_data, 'city_traffic_networks.pkl'), 'rb') as f:
    g_world_city_traffic_network = pickle.load(f)
    pass  # with

# %%
# 可视化城市交通网络
world_city_pos = dict([(id_city_node, g_world_city_traffic_network.nodes[id_city_node]['pos']) for id_city_node in g_world_city_traffic_network.nodes])

options_world_city_traffic_network = {
    'node_color': 'white',  # 节点颜色
    'linewidths': 2,  # 节点边框宽度
    'edgecolors': 'black',  # 节点边框颜色
    'node_size': 300,  # 节点大小
    'width': 2,  # 边宽度
    'with_labels': True,  # 是否显示节点标签
    'font_size': 18,  # 节点标签字体大小
    'font_color': 'black',  # 节点标签字体颜色
}
nx.draw_networkx(g_world_city_traffic_network, pos=world_city_pos, **options_world_city_traffic_network)

plt.figure(1, figsize=(4, 4))
plt.axis('off')
ax = plt.gca()
ax.set_aspect('equal')
ax.set_title('全国交通网络')
plt.show()

# # 标记路径边
# print(shortest_path)
# shortest_path_for_vis = list(zip(shortest_path[:-1], shortest_path[1:]))
#
# # 标记路径颜色为黄色
# edge_colors = [
#     'yellow' if edge in zip(shortest_path[:-1], shortest_path[1:]) else 'black' for edge in g_world_city_traffic_network.edges()
# ]
#
# nx.draw(g_world_city_traffic_network, nodelist=[world_city_node_start, world_city_node_end], pos=pos)
# plt.show()
