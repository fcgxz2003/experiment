import pandas as pd
from collections import defaultdict
import numpy as np


def read_nt():
    df = pd.read_csv("dataset/networktopology.csv", sep=',', header=None,
                     names=['nt_id', 'src_id', 'src_type', 'src_hostname', 'src_ip', 'src_port',
                            'src_network_TCPConnectionCount', 'src_network_UploadTCPConnectionCount',
                            'src_network_location', 'src_network_idc',

                            'dest_id_1', 'dest_type_1', 'dest_hostname_1', 'dest_ip_1', 'dest_port_1',
                            'dest_network_TCPConnectionCount_1', 'dest_network_UploadTCPConnectionCount_1',
                            'dest_network_location_1', 'dest_network_idc_1', 'averageRTT1', 'createdAt1', 'updatedAt1',

                            'dest_id_2', 'dest_type_2', 'dest_hostname_2', 'dest_ip_2', 'dest_port_2',
                            'dest_network_TCPConnectionCount_2', 'dest_network_UploadTCPConnectionCount_2',
                            'dest_network_location_2', 'dest_network_idc_2', 'averageRTT2', 'createdAt2', 'updatedAt2',

                            'dest_id_3', 'dest_type_3', 'dest_hostname_3', 'dest_ip_3', 'dest_port_3',
                            'dest_network_TCPConnectionCount_3', 'dest_network_UploadTCPConnectionCount_3',
                            'dest_network_location_3', 'dest_network_idc_3', 'averageRTT3', 'createdAt3', 'updatedAt3',

                            'dest_id_4', 'dest_type_4', 'dest_hostname_4', 'dest_ip_4', 'dest_port_4',
                            'dest_network_TCPConnectionCount_4', 'dest_network_UploadTCPConnectionCount_4',
                            'dest_network_location_4', 'dest_network_idc_4', 'averageRTT4', 'createdAt4', 'updatedAt4',

                            'dest_id_5', 'dest_type_5', 'dest_hostname_5', 'dest_ip_5', 'dest_port_5',
                            'dest_network_TCPConnectionCount_5', 'dest_network_UploadTCPConnectionCount_5',
                            'dest_network_location_5', 'dest_network_idc_5', 'averageRTT5', 'createdAt5', 'updatedAt5',

                            'nt_createdAt'
                            ])

    # 构建host id 对应的节点下标。
    node_index = {}
    i = 0

    # 特征工程
    raw_feature = []
    # 遍历csv, 将host_id 转成index
    for index in df.index:
        src_id = df.loc[index]['src_id']
        dest_id_1 = df.loc[index]['dest_id_1']
        dest_id_2 = df.loc[index]['dest_id_2']
        dest_id_3 = df.loc[index]['dest_id_3']
        dest_id_4 = df.loc[index]['dest_id_4']
        dest_id_5 = df.loc[index]['dest_id_5']

        if src_id != "" and (node_index.get(src_id) is None):
            node_index[src_id] = i
            src_ip = df.loc[index]['src_ip']
            src_network_location = df.loc[index]['src_network_location']
            src_network_idc = df.loc[index]['src_network_idc']
            raw_feature.append([src_ip, src_network_location, src_network_idc])
            i = i + 1

        if dest_id_1 != "" and (node_index.get(dest_id_1) is None):
            node_index[dest_id_1] = i
            dest_ip_1 = df.loc[index]['dest_ip_1']
            dest_network_location_1 = df.loc[index]['dest_network_location_1']
            dest_network_idc_1 = df.loc[index]['dest_network_idc_1']
            raw_feature.append([dest_ip_1, dest_network_location_1, dest_network_idc_1])
            i = i + 1

        if dest_id_2 != "" and (node_index.get(dest_id_2) is None):
            node_index[dest_id_2] = i
            dest_ip_2 = df.loc[index]['dest_ip_2']
            dest_network_location_2 = df.loc[index]['dest_network_location_2']
            dest_network_idc_2 = df.loc[index]['dest_network_idc_2']
            raw_feature.append([dest_ip_2, dest_network_location_2, dest_network_idc_2])
            i = i + 1

        if dest_id_3 != "" and (node_index.get(dest_id_3) is None):
            node_index[dest_id_3] = i
            dest_ip_3 = df.loc[index]['dest_ip_3']
            dest_network_location_3 = df.loc[index]['dest_network_location_3']
            dest_network_idc_3 = df.loc[index]['dest_network_idc_3']
            raw_feature.append([dest_ip_3, dest_network_location_3, dest_network_idc_3])
            i = i + 1

        if dest_id_4 != "" and (node_index.get(dest_id_4) is None):
            node_index[dest_id_4] = i
            dest_ip_4 = df.loc[index]['dest_ip_4']
            dest_network_location_4 = df.loc[index]['dest_network_location_4']
            dest_network_idc_4 = df.loc[index]['dest_network_idc_4']
            raw_feature.append([dest_ip_4, dest_network_location_4, dest_network_idc_4])
            i = i + 1

        if dest_id_5 != "" and (node_index.get(dest_id_5) is None):
            node_index[dest_id_5] = i
            dest_ip_5 = df.loc[index]['dest_ip_5']
            dest_network_location_5 = df.loc[index]['dest_network_location_5']
            dest_network_idc_5 = df.loc[index]['dest_network_idc_5']
            raw_feature.append([dest_ip_5, dest_network_location_5, dest_network_idc_5])
            i = i + 1

    print(node_index)

    # 构建邻居信息表 1：[2,3,4]...
    adj_lists = defaultdict(set)
    # 遍历csv, 将host_id 转成index
    for index in df.index:
        src_id = df.loc[index]['src_id']
        dest_id_1 = df.loc[index]['dest_id_1']
        dest_id_2 = df.loc[index]['dest_id_2']
        dest_id_3 = df.loc[index]['dest_id_3']
        dest_id_4 = df.loc[index]['dest_id_4']
        dest_id_5 = df.loc[index]['dest_id_5']

        src_id = node_index[src_id]
        adj_lists[src_id].add(node_index[dest_id_1])
        adj_lists[src_id].add(node_index[dest_id_2])
        adj_lists[src_id].add(node_index[dest_id_3])
        adj_lists[src_id].add(node_index[dest_id_4])
        adj_lists[src_id].add(node_index[dest_id_5])

    adj_lists = {k: np.array(list(v)) for k, v in adj_lists.items()}
    print(adj_lists)

    print(raw_feature)

if __name__ == '__main__':
    read_nt()
