# %% md
# # 生成简易的分级分层的道路交通网络地图（#HACK 未适配，未开发，不能用）
# 
# %% md
# 有一种全国交通层次网络结构。假设在一段时间内是静态的（不修路），是双向的，是无环的，权重是正的，表示路径长度。全国的各个城市视为节点，通过覆盖全国的交通网络连接，这是个顶级网络。每一个城内部又是一个个的城内交通网络，暨城市的每一个道路交叉点视为一个节点，形成网络，这是低级网络。
# 
# %% md
# 设定：
# 
# - 假设不考虑火车、飞机，只考虑高速公路网络。
# - 设定对于低级网络，暨城内部交通网络节点集合，包括两类，一类是城内部节点集合，一类是沟通其他城市的高速出入口节点集合。顶级网络当中，该城市节点的的邻居，对应这该城市有相同个数的高速出入口节点。这些高速出入口节点分别与顶级网络节点连边对应
# - 假设顶级网络（高速公路网络）的通行效率远远高于城内部的通行效率。
# - 假设城市在顶级网络视为一个高速公路节点。不需要进入城内部的道路网络。路过中间城市的时候，而仅仅直接走顶级网络G在该城市的高速节点$G_j$，暨绕城高速节点，而不需要出入该城市j的高速出入口节点进入城内部。
# 
# %% md
# 对于城内部的网络来说，按照是否路由节点划分，有两种类型：第一种是关键节点，也就是计算用的节点。另一种是路由节点，只是为了在关键节点之间做曲线连边插值的。关键节点按照内外部划分，有两种类型的节点：内部节点和外部节点。内部节点是不和其他城市网络连接的节点。外部节点就是高速公路出入口节点，与上一层次的网络的城市节点$G_i$对接。通过该类节点对应到$G_i$，从而连接到全国路网。一个城市的外部节点数量等于上一层次的网络的城市节点$G_i$的邻居节点数量。一个城市的外部节点与上一层次的网络的城市节点$G_i$的邻居节点一一对应。从交通网络角度来理解，就是每一个城市的每一个高速公路出入口一一对应其相邻城市。
# 
# 每一个城市 i 内部有自己的城市道路网络 $C_i$（低级）。全国高速是一个顶级网络 G，每一个节点是该城市的高速节点。全国高速在路过其他城市的时候，可以不进入城市，而是仅仅走高速节点。只有在出入城市的时候才涉及到城市道路网络 $C_i$。其中出入城市的道路网络与高速节点的连接是通过高速路口节点连接的。$C_i$的出入口节点就是通过G连接城市的邻居城市的节点集合。
# 
# 假设从$i=城市A$到$i=城市B$，第一步要经过$城市C$，最后一步是从$城市E$到$城市B$，那么规划的时候，$A$的出口节点只能是$A$到$C$的连接点，$B$的入口节点只能是$E$到$B$的连接点。也就是说，先在网络G规划A到B的路径$p1$。
# 
# 举例，假设$p1$的第一个节点是A，第二个节点是C，倒数第二个节点是E，最后一个节点是B。然后根据规划后的路径，再规划A内部的起点节点a1到高速出口节点$a_out_C$的路径$p2$，以及规划节点E和节点B的连接的高速入口节点$b_in_E$到城市B内部终点$b3$节点的路径。
# 
# 出入口节点其实在网络中仅仅根据路径的进出才区分了。好比说我要去东边的城市，就只能走东边的高速出口，而不能走西边的高速出口，因为从西边的高速出口走，就会走到别的城市，这样就走错路了。对于城市来说，离开的城市的高速出入口节点只和接下来一步要路过的邻居城市的顶级网络G对应的节点连接。进入的城市的高速出入口节点只和上一步路过的邻居城市的顶级网络G对应的节点连接。
# %% md
# # 开发实现
# %%
import logging, pickle
from pathlib import Path
import networkx as nx
import numpy as np
from complex_network_operation.functions.generations.fun_generate_points import generate_points
from complex_network_operation.functions.generations.fun_generate_roads_map import generate_roadsNetwork_by_machanism
from complex_network_operation.utils.tools import Tools

logging.basicConfig(level=logging.INFO)
# %%

# %%
# 参数设置
max_levels = 10  # 最大层级数
world_circle_radius = 75
world_circle_density_distance = 50
city_nodes_density_distance = 2
city_circle_radius = 3


# 递归生成节点位置
def generate_nodes(level, origin, radius, density_distance):
    if level > max_levels:
        return []
    nodes_pos = generate_points(
        set_densityDistance=density_distance,
        set_numPoints=None,
        set_circleRadius=radius,
        circle_origin=origin,
        distribution='Posssion Disk'
    )
    sub_nodes = []
    for pos in nodes_pos:
        sub_nodes.extend(generate_nodes(level + 1, pos, city_circle_radius, city_nodes_density_distance))
    return nodes_pos + sub_nodes


# 递归生成交通网络
def generate_network(level, nodes_pos):
    if level > max_levels:
        return None
    g_network, _ = generate_roadsNetwork_by_machanism(
        nodes_pos=nodes_pos,
        set_num_interpolated_density_distance=1.0,
        is_preview_plot=False
    )
    for node in g_network.nodes:
        sub_nodes_pos = generate_nodes(level + 1, g_network.nodes[node]['pos'], city_circle_radius, city_nodes_density_distance)
        sub_network = generate_network(level + 1, sub_nodes_pos)
        g_network.nodes[node]['sub_network'] = sub_network
    return g_network


# 生成顶级网络
city_nodes_pos = generate_nodes(1, np.array([0, 0]), world_circle_radius, world_circle_density_distance)
g_world_city_traffic_network = generate_network(1, city_nodes_pos)


# 添加节点和边属性
def add_attributes(g_network, level):
    if level > max_levels or g_network is None:
        return
    m = len(str(g_network.number_of_nodes()))
    for node in g_network.nodes:
        g_network.nodes[node]['name'] = f"节点={node:0{m}d}"
        g_network.nodes[node]['type'] = f"第{level}级节点"
        add_attributes(g_network.nodes[node].get('sub_network'), level + 1)
    m = len(str(g_network.number_of_edges()))
    for edge in g_network.edges:
        g_network.edges[edge]['name'] = f"边={edge[0]:0{m}d}-{edge[1]:0{m}d}"
        g_network.edges[edge]['type'] = f"第{level}级边"


add_attributes(g_world_city_traffic_network, 1)


# 选取出入口节点
def select_io_nodes(g_network, level):
    if level > max_levels or g_network is None:
        return
    for node in g_network.nodes:
        sub_network = g_network.nodes[node].get('sub_network')
        if sub_network:
            ids_neighbors = list(g_network.neighbors(node))
            city_io_nodes = dict()
            for id_neighbor in ids_neighbors:
                min_distance = np.inf
                city_io_node = None
                for sub_node in sub_network.nodes:
                    distance = np.linalg.norm(sub_network.nodes[sub_node]['pos'] - g_network.nodes[id_neighbor]['pos'])
                    if distance < min_distance:
                        min_distance = distance
                        city_io_node = sub_node
                city_io_nodes[id_neighbor] = city_io_node
                sub_network.nodes[city_io_node]['type'] = '城市出入口'
                sub_network.nodes[city_io_node]['name'] = f"城市出入口={city_io_node:0{len(str(sub_network.number_of_nodes()))}d}-{id_neighbor:0{len(str(g_network.number_of_nodes()))}d}"
            sub_network.graph['city_io_nodes'] = city_io_nodes
            select_io_nodes(sub_network, level + 1)


select_io_nodes(g_world_city_traffic_network, 1)

# 保存生成的城市交通网络
folderpath_project = Path(Tools.get_project_rootpath('ComplexNetworkOperation', '.')).resolve()
folderpath_output_data = folderpath_project / 'data/分级道路交通网络地图'
with open(Path(folderpath_output_data, 'city_traffic_networks.pkl'), 'wb') as f:
    pickle.dump(g_world_city_traffic_network, f)
