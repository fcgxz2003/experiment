import ipaddress

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

        # # 现在是有向图，之后可以试试无向图
        # src_id = node_index[src_id]
        # adj_lists[src_id].add(node_index[dest_id_1])
        # adj_lists[src_id].add(node_index[dest_id_2])
        # adj_lists[src_id].add(node_index[dest_id_3])
        # adj_lists[src_id].add(node_index[dest_id_4])
        # adj_lists[src_id].add(node_index[dest_id_5])

        # 无向图
        src_id = node_index[src_id]
        adj_lists[src_id].add(node_index[dest_id_1])
        adj_lists[src_id].add(node_index[dest_id_2])
        adj_lists[src_id].add(node_index[dest_id_3])
        adj_lists[src_id].add(node_index[dest_id_4])
        adj_lists[src_id].add(node_index[dest_id_5])

        adj_lists[node_index[dest_id_1]].add(src_id)
        adj_lists[node_index[dest_id_2]].add(src_id)
        adj_lists[node_index[dest_id_3]].add(src_id)
        adj_lists[node_index[dest_id_4]].add(src_id)
        adj_lists[node_index[dest_id_5]].add(src_id)


    node_num = i
    adj_lists = {k: np.array(list(v)) for k, v in adj_lists.items()}
    print(node_num)
    print(adj_lists)
    print(raw_feature)

    # ip 用32维度，看效果
    # location 先默认是2*3*3 = 18 维度  省| 市 | 县
    # idc 先默认只有10个idc
    feature = []

    for f in raw_feature:
        ip = ip_to_binary(f[0])
        if pd.isna(f[1]):
            ip = ip + [0 for i in range(18)]
        if pd.isna(f[2]):
            ip = ip + [0 for i in range(10)]

        feature.append(ip)

    print(feature)
    return node_num, feature, adj_lists, node_index


def ip_to_binary(ip_address):
    # 将IP地址解析为IPv4Network对象
    network = ipaddress.IPv4Network(ip_address)
    # 获取网络地址的整数表示
    ip_integer = int(network.network_address)
    # 将整数转换为32位二进制列表
    binary_list = [int(bit) for bit in f'{ip_integer:032b}']
    return binary_list


def read_d(node_index):
    df = pd.read_csv("dataset/download.csv", sep=',', header=None,
                     names=[
                         'peer_id', 'peer_tag', 'peer_application', 'peer_state',
                         'error_code', 'error_message',
                         'cost', 'finishedPieceCount',

                         'task_id', 'task_url', 'task_type', 'task_contentLength', 'task_TotalPieceCount',
                         'task_BackToSourceLimit', 'task_BackToSourcePeerCount', 'task_state', 'task_CreatedAt',
                         'task_UpdatedAt',

                         'host_id', 'host_type', 'host_hostname', 'host_ip', 'host_port', 'host_downloadPort',
                         'host_OS', 'host_platform', 'host_PlatformFamily', 'host_PlatformVersion',
                         'host_KernelVersion', 'host_ConcurrentUploadLimit', 'host_ConcurrentUploadCount',
                         'host_UploadCount', 'host_UploadFailedCount',

                         'host_cpu_LogicalCount', 'host_cpu_PhysicalCount', 'host_cpu_Percent',
                         'host_cpu_ProcessPercent',

                         'host_cpu_cputime_User', 'host_cpu_cputime_System', 'host_cpu_cputime_Idle',
                         'host_cpu_cputime_Nice', 'host_cpu_cputime_Iowait', 'host_cpu_cputime_Irq',
                         'host_cpu_cputime_Softirq', 'host_cpu_cputime_Steal', 'host_cpu_cputime_Guest',
                         'host_cpu_cputime_GuestNice',

                         'host_cpu_memory_Total', 'host_cpu_memory_Available', 'host_cpu_memory_Used',
                         'host_cpu_memory_UsedPercent', 'host_cpu_memory_ProcessUsedPercent', 'host_cpu_memory_Free',

                         'host_cpu_network_TCPConnectionCount', 'host_cpu_network_UploadTCPConnectionCount',
                         'host_cpu_network_Location', 'host_cpu_network_IDC',

                         'host_cpu_disk_Total', 'host_cpu_disk_Free', 'host_cpu_disk_Used', 'host_cpu_disk_UsedPercent',
                         'host_cpu_disk_InodesTotal', 'host_cpu_disk_InodesUsed', 'host_cpu_disk_InodesFree',
                         'host_cpu_disk_InodesUsedPercent',

                         'host_cpu_build_GitVersion', 'host_cpu_build_GitCommit', 'host_cpu_build_GoVersion',
                         'host_cpu_build_Platform',

                         'host_SchedulerClusterID', 'host_CreatedAt', 'host_UpdatedAt',

                         # --------------------------------
                         # parent 0
                         '0parent_id', '0parent_Tag', '0parent_Application', '0parent_State', '0parent_Cost',
                         '0parent_UploadPieceCount', '0parent_FinishedPieceCount',

                         '0parent_host_id', '0parent_host_type', '0parent_host_hostname', '0parent_host_ip',
                         '0parent_host_port', '0parent_host_downloadPort',
                         '0parent_host_OS', '0parent_host_platform', '0parent_host_PlatformFamily',
                         '0parent_host_PlatformVersion',
                         '0parent_host_KernelVersion', '0parent_host_ConcurrentUploadLimit',
                         '0parent_host_ConcurrentUploadCount',
                         '0parent_host_UploadCount', '0parent_host_UploadFailedCount',

                         '0parent_host_cpu_LogicalCount', '0parent_host_cpu_PhysicalCount', '0parent_host_cpu_Percent',
                         '0parent_host_cpu_ProcessPercent',

                         '0parent_host_cpu_cputime_User', '0parent_host_cpu_cputime_System',
                         '0parent_host_cpu_cputime_Idle',
                         '0parent_host_cpu_cputime_Nice', '0parent_host_cpu_cputime_Iowait',
                         '0parent_host_cpu_cputime_Irq',
                         '0parent_host_cpu_cputime_Softirq', '0parent_host_cpu_cputime_Steal',
                         '0parent_host_cpu_cputime_Guest',
                         '0parent_host_cpu_cputime_GuestNice',

                         '0parent_host_cpu_memory_Total', '0parent_host_cpu_memory_Available',
                         '0parent_host_cpu_memory_Used',
                         '0parent_host_cpu_memory_UsedPercent', '0parent_host_cpu_memory_ProcessUsedPercent',
                         '0parent_host_cpu_memory_Free',

                         '0parent_host_cpu_network_TCPConnectionCount',
                         '0parent_host_cpu_network_UploadTCPConnectionCount',
                         '0parent_host_cpu_network_Location', '0parent_host_cpu_network_IDC',

                         '0parent_host_cpu_disk_Total', '0parent_host_cpu_disk_Free', '0parent_host_cpu_disk_Used',
                         '0parent_host_cpu_disk_UsedPercent',
                         '0parent_host_cpu_disk_InodesTotal', '0parent_host_cpu_disk_InodesUsed',
                         '0parent_host_cpu_disk_InodesFree',
                         '0parent_host_cpu_disk_InodesUsedPercent',

                         '0parent_host_cpu_build_GitVersion', '0parent_host_cpu_build_GitCommit',
                         '0parent_host_cpu_build_GoVersion',
                         '0parent_host_cpu_build_Platform',

                         '0parent_host_SchedulerClusterID', '0parent_host_CreatedAt', '0parent_host_UpdatedAt',

                         # parent 0 piece
                         '0parent_0piece_Length', '0parent_0piece_cost', '0parent_0piece_createdAt',
                         '0parent_1piece_Length', '0parent_1piece_cost', '0parent_1piece_createdAt',
                         '0parent_2piece_Length', '0parent_2piece_cost', '0parent_2piece_createdAt',
                         '0parent_3piece_Length', '0parent_3piece_cost', '0parent_3piece_createdAt',
                         '0parent_4piece_Length', '0parent_4piece_cost', '0parent_4piece_createdAt',
                         '0parent_5piece_Length', '0parent_5piece_cost', '0parent_5piece_createdAt',
                         '0parent_6piece_Length', '0parent_6piece_cost', '0parent_6piece_createdAt',
                         '0parent_7piece_Length', '0parent_7piece_cost', '0parent_7piece_createdAt',
                         '0parent_8piece_Length', '0parent_8piece_cost', '0parent_8piece_createdAt',
                         '0parent_9piece_Length', '0parent_9piece_cost', '0parent_9piece_createdAt',

                         '0parent_createdAt', '0parent_updatedAt',
                         # --------------------------------
                         # parent 1
                         '1parent_id', '1parent_Tag', '1parent_Application', '1parent_State', '1parent_Cost',
                         '1parent_UploadPieceCount', '1parent_FinishedPieceCount',

                         '1parent_host_id', '1parent_host_type', '1parent_host_hostname', '1parent_host_ip',
                         '1parent_host_port', '1parent_host_downloadPort',
                         '1parent_host_OS', '1parent_host_platform', '1parent_host_PlatformFamily',
                         '1parent_host_PlatformVersion',
                         '1parent_host_KernelVersion', '1parent_host_ConcurrentUploadLimit',
                         '1parent_host_ConcurrentUploadCount',
                         '1parent_host_UploadCount', '1parent_host_UploadFailedCount',

                         '1parent_host_cpu_LogicalCount', '1parent_host_cpu_PhysicalCount', '1parent_host_cpu_Percent',
                         '1parent_host_cpu_ProcessPercent',

                         '1parent_host_cpu_cputime_User', '1parent_host_cpu_cputime_System',
                         '1parent_host_cpu_cputime_Idle',
                         '1parent_host_cpu_cputime_Nice', '1parent_host_cpu_cputime_Iowait',
                         '1parent_host_cpu_cputime_Irq',
                         '1parent_host_cpu_cputime_Softirq', '1parent_host_cpu_cputime_Steal',
                         '1parent_host_cpu_cputime_Guest',
                         '1parent_host_cpu_cputime_GuestNice',

                         '1parent_host_cpu_memory_Total', '1parent_host_cpu_memory_Available',
                         '1parent_host_cpu_memory_Used',
                         '1parent_host_cpu_memory_UsedPercent', '1parent_host_cpu_memory_ProcessUsedPercent',
                         '1parent_host_cpu_memory_Free',

                         '1parent_host_cpu_network_TCPConnectionCount',
                         '1parent_host_cpu_network_UploadTCPConnectionCount',
                         '1parent_host_cpu_network_Location', '1parent_host_cpu_network_IDC',

                         '1parent_host_cpu_disk_Total', '1parent_host_cpu_disk_Free', '1parent_host_cpu_disk_Used',
                         '1parent_host_cpu_disk_UsedPercent',
                         '1parent_host_cpu_disk_InodesTotal', '1parent_host_cpu_disk_InodesUsed',
                         '1parent_host_cpu_disk_InodesFree',
                         '1parent_host_cpu_disk_InodesUsedPercent',

                         '1parent_host_cpu_build_GitVersion', '1parent_host_cpu_build_GitCommit',
                         '1parent_host_cpu_build_GoVersion',
                         '1parent_host_cpu_build_Platform',

                         '1parent_host_SchedulerClusterID', '1parent_host_CreatedAt', '1parent_host_UpdatedAt',

                         # parent 0 piece
                         '1parent_0piece_Length', '1parent_0piece_cost', '1parent_0piece_createdAt',
                         '1parent_1piece_Length', '1parent_1piece_cost', '1parent_1piece_createdAt',
                         '1parent_2piece_Length', '1parent_2piece_cost', '1parent_2piece_createdAt',
                         '1parent_3piece_Length', '1parent_3piece_cost', '1parent_3piece_createdAt',
                         '1parent_4piece_Length', '1parent_4piece_cost', '1parent_4piece_createdAt',
                         '1parent_5piece_Length', '1parent_5piece_cost', '1parent_5piece_createdAt',
                         '1parent_6piece_Length', '1parent_6piece_cost', '1parent_6piece_createdAt',
                         '1parent_7piece_Length', '1parent_7piece_cost', '1parent_7piece_createdAt',
                         '1parent_8piece_Length', '1parent_8piece_cost', '1parent_8piece_createdAt',
                         '1parent_9piece_Length', '1parent_9piece_cost', '1parent_9piece_createdAt',

                         '1parent_createdAt', '1parent_updatedAt',
                         # --------------------------------
                         # parent 2
                         '2parent_id', '2parent_Tag', '2parent_Application', '2parent_State', '2parent_Cost',
                         '2parent_UploadPieceCount', '2parent_FinishedPieceCount',

                         '2parent_host_id', '2parent_host_type', '2parent_host_hostname', '2parent_host_ip',
                         '2parent_host_port', '2parent_host_downloadPort',
                         '2parent_host_OS', '2parent_host_platform', '2parent_host_PlatformFamily',
                         '2parent_host_PlatformVersion',
                         '2parent_host_KernelVersion', '2parent_host_ConcurrentUploadLimit',
                         '2parent_host_ConcurrentUploadCount',
                         '2parent_host_UploadCount', '2parent_host_UploadFailedCount',

                         '2parent_host_cpu_LogicalCount', '2parent_host_cpu_PhysicalCount', '2parent_host_cpu_Percent',
                         '2parent_host_cpu_ProcessPercent',

                         '2parent_host_cpu_cputime_User', '2parent_host_cpu_cputime_System',
                         '2parent_host_cpu_cputime_Idle',
                         '2parent_host_cpu_cputime_Nice', '2parent_host_cpu_cputime_Iowait',
                         '2parent_host_cpu_cputime_Irq',
                         '2parent_host_cpu_cputime_Softirq', '2parent_host_cpu_cputime_Steal',
                         '2parent_host_cpu_cputime_Guest',
                         '2parent_host_cpu_cputime_GuestNice',

                         '2parent_host_cpu_memory_Total', '2parent_host_cpu_memory_Available',
                         '2parent_host_cpu_memory_Used',
                         '2parent_host_cpu_memory_UsedPercent', '2parent_host_cpu_memory_ProcessUsedPercent',
                         '2parent_host_cpu_memory_Free',

                         '2parent_host_cpu_network_TCPConnectionCount',
                         '2parent_host_cpu_network_UploadTCPConnectionCount',
                         '2parent_host_cpu_network_Location', '2parent_host_cpu_network_IDC',

                         '2parent_host_cpu_disk_Total', '2parent_host_cpu_disk_Free', '2parent_host_cpu_disk_Used',
                         '2parent_host_cpu_disk_UsedPercent',
                         '2parent_host_cpu_disk_InodesTotal', '2parent_host_cpu_disk_InodesUsed',
                         '2parent_host_cpu_disk_InodesFree',
                         '2parent_host_cpu_disk_InodesUsedPercent',

                         '2parent_host_cpu_build_GitVersion', '2parent_host_cpu_build_GitCommit',
                         '2parent_host_cpu_build_GoVersion',
                         '2parent_host_cpu_build_Platform',

                         '2parent_host_SchedulerClusterID', '2parent_host_CreatedAt', '2parent_host_UpdatedAt',

                         # parent 0 piece
                         '2parent_0piece_Length', '2parent_0piece_cost', '2parent_0piece_createdAt',
                         '2parent_1piece_Length', '2parent_1piece_cost', '2parent_1piece_createdAt',
                         '2parent_2piece_Length', '2parent_2piece_cost', '2parent_2piece_createdAt',
                         '2parent_3piece_Length', '2parent_3piece_cost', '2parent_3piece_createdAt',
                         '2parent_4piece_Length', '2parent_4piece_cost', '2parent_4piece_createdAt',
                         '2parent_5piece_Length', '2parent_5piece_cost', '2parent_5piece_createdAt',
                         '2parent_6piece_Length', '2parent_6piece_cost', '2parent_6piece_createdAt',
                         '2parent_7piece_Length', '2parent_7piece_cost', '2parent_7piece_createdAt',
                         '2parent_8piece_Length', '2parent_8piece_cost', '2parent_8piece_createdAt',
                         '2parent_9piece_Length', '2parent_9piece_cost', '2parent_9piece_createdAt',

                         '2parent_createdAt', '2parent_updatedAt',
                         # --------------------------------
                         # parent 3
                         '3parent_id', '3parent_Tag', '3parent_Application', '3parent_State', '3parent_Cost',
                         '3parent_UploadPieceCount', '3parent_FinishedPieceCount',

                         '3parent_host_id', '3parent_host_type', '3parent_host_hostname', '3parent_host_ip',
                         '3parent_host_port', '3parent_host_downloadPort',
                         '3parent_host_OS', '3parent_host_platform', '3parent_host_PlatformFamily',
                         '3parent_host_PlatformVersion',
                         '3parent_host_KernelVersion', '3parent_host_ConcurrentUploadLimit',
                         '3parent_host_ConcurrentUploadCount',
                         '3parent_host_UploadCount', '3parent_host_UploadFailedCount',

                         '3parent_host_cpu_LogicalCount', '3parent_host_cpu_PhysicalCount', '3parent_host_cpu_Percent',
                         '3parent_host_cpu_ProcessPercent',

                         '3parent_host_cpu_cputime_User', '3parent_host_cpu_cputime_System',
                         '3parent_host_cpu_cputime_Idle',
                         '3parent_host_cpu_cputime_Nice', '3parent_host_cpu_cputime_Iowait',
                         '3parent_host_cpu_cputime_Irq',
                         '3parent_host_cpu_cputime_Softirq', '3parent_host_cpu_cputime_Steal',
                         '3parent_host_cpu_cputime_Guest',
                         '3parent_host_cpu_cputime_GuestNice',

                         '3parent_host_cpu_memory_Total', '3parent_host_cpu_memory_Available',
                         '3parent_host_cpu_memory_Used',
                         '3parent_host_cpu_memory_UsedPercent', '3parent_host_cpu_memory_ProcessUsedPercent',
                         '3parent_host_cpu_memory_Free',

                         '3parent_host_cpu_network_TCPConnectionCount',
                         '3parent_host_cpu_network_UploadTCPConnectionCount',
                         '3parent_host_cpu_network_Location', '3parent_host_cpu_network_IDC',

                         '3parent_host_cpu_disk_Total', '3parent_host_cpu_disk_Free', '3parent_host_cpu_disk_Used',
                         '3parent_host_cpu_disk_UsedPercent',
                         '3parent_host_cpu_disk_InodesTotal', '3parent_host_cpu_disk_InodesUsed',
                         '3parent_host_cpu_disk_InodesFree',
                         '3parent_host_cpu_disk_InodesUsedPercent',

                         '3parent_host_cpu_build_GitVersion', '3parent_host_cpu_build_GitCommit',
                         '3parent_host_cpu_build_GoVersion',
                         '3parent_host_cpu_build_Platform',

                         '3parent_host_SchedulerClusterID', '3parent_host_CreatedAt', '3parent_host_UpdatedAt',

                         # parent 0 piece
                         '3parent_0piece_Length', '3parent_0piece_cost', '3parent_0piece_createdAt',
                         '3parent_1piece_Length', '3parent_1piece_cost', '3parent_1piece_createdAt',
                         '3parent_2piece_Length', '3parent_2piece_cost', '3parent_2piece_createdAt',
                         '3parent_3piece_Length', '3parent_3piece_cost', '3parent_3piece_createdAt',
                         '3parent_4piece_Length', '3parent_4piece_cost', '3parent_4piece_createdAt',
                         '3parent_5piece_Length', '3parent_5piece_cost', '3parent_5piece_createdAt',
                         '3parent_6piece_Length', '3parent_6piece_cost', '3parent_6piece_createdAt',
                         '3parent_7piece_Length', '3parent_7piece_cost', '3parent_7piece_createdAt',
                         '3parent_8piece_Length', '3parent_8piece_cost', '3parent_8piece_createdAt',
                         '3parent_9piece_Length', '3parent_9piece_cost', '3parent_9piece_createdAt',

                         '3parent_createdAt', '3parent_updatedAt',
                         # --------------------------------
                         # parent 4
                         '4parent_id', '4parent_Tag', '4parent_Application', '4parent_State', '4parent_Cost',
                         '4parent_UploadPieceCount', '4parent_FinishedPieceCount',

                         '4parent_host_id', '4parent_host_type', '4parent_host_hostname', '4parent_host_ip',
                         '4parent_host_port', '4parent_host_downloadPort',
                         '4parent_host_OS', '4parent_host_platform', '4parent_host_PlatformFamily',
                         '4parent_host_PlatformVersion',
                         '4parent_host_KernelVersion', '4parent_host_ConcurrentUploadLimit',
                         '4parent_host_ConcurrentUploadCount',
                         '4parent_host_UploadCount', '4parent_host_UploadFailedCount',

                         '4parent_host_cpu_LogicalCount', '4parent_host_cpu_PhysicalCount', '4parent_host_cpu_Percent',
                         '4parent_host_cpu_ProcessPercent',

                         '4parent_host_cpu_cputime_User', '4parent_host_cpu_cputime_System',
                         '4parent_host_cpu_cputime_Idle',
                         '4parent_host_cpu_cputime_Nice', '4parent_host_cpu_cputime_Iowait',
                         '4parent_host_cpu_cputime_Irq',
                         '4parent_host_cpu_cputime_Softirq', '4parent_host_cpu_cputime_Steal',
                         '4parent_host_cpu_cputime_Guest',
                         '4parent_host_cpu_cputime_GuestNice',

                         '4parent_host_cpu_memory_Total', '4parent_host_cpu_memory_Available',
                         '4parent_host_cpu_memory_Used',
                         '4parent_host_cpu_memory_UsedPercent', '4parent_host_cpu_memory_ProcessUsedPercent',
                         '4parent_host_cpu_memory_Free',

                         '4parent_host_cpu_network_TCPConnectionCount',
                         '4parent_host_cpu_network_UploadTCPConnectionCount',
                         '4parent_host_cpu_network_Location', '4parent_host_cpu_network_IDC',

                         '4parent_host_cpu_disk_Total', '4parent_host_cpu_disk_Free', '4parent_host_cpu_disk_Used',
                         '4parent_host_cpu_disk_UsedPercent',
                         '4parent_host_cpu_disk_InodesTotal', '4parent_host_cpu_disk_InodesUsed',
                         '4parent_host_cpu_disk_InodesFree',
                         '4parent_host_cpu_disk_InodesUsedPercent',

                         '4parent_host_cpu_build_GitVersion', '4parent_host_cpu_build_GitCommit',
                         '4parent_host_cpu_build_GoVersion',
                         '4parent_host_cpu_build_Platform',

                         '4parent_host_SchedulerClusterID', '4parent_host_CreatedAt', '4parent_host_UpdatedAt',

                         # parent 0 piece
                         '4parent_0piece_Length', '4parent_0piece_cost', '4parent_0piece_createdAt',
                         '4parent_1piece_Length', '4parent_1piece_cost', '4parent_1piece_createdAt',
                         '4parent_2piece_Length', '4parent_2piece_cost', '4parent_2piece_createdAt',
                         '4parent_3piece_Length', '4parent_3piece_cost', '4parent_3piece_createdAt',
                         '4parent_4piece_Length', '4parent_4piece_cost', '4parent_4piece_createdAt',
                         '4parent_5piece_Length', '4parent_5piece_cost', '4parent_5piece_createdAt',
                         '4parent_6piece_Length', '4parent_6piece_cost', '4parent_6piece_createdAt',
                         '4parent_7piece_Length', '4parent_7piece_cost', '4parent_7piece_createdAt',
                         '4parent_8piece_Length', '4parent_8piece_cost', '4parent_8piece_createdAt',
                         '4parent_9piece_Length', '4parent_9piece_cost', '4parent_9piece_createdAt',

                         '4parent_createdAt', '4parent_updatedAt',
                         # --------------------------------
                         # parent 5
                         '5parent_id', '5parent_Tag', '5parent_Application', '5parent_State', '5parent_Cost',
                         '5parent_UploadPieceCount', '5parent_FinishedPieceCount',

                         '5parent_host_id', '5parent_host_type', '5parent_host_hostname', '5parent_host_ip',
                         '5parent_host_port', '5parent_host_downloadPort',
                         '5parent_host_OS', '5parent_host_platform', '5parent_host_PlatformFamily',
                         '5parent_host_PlatformVersion',
                         '5parent_host_KernelVersion', '5parent_host_ConcurrentUploadLimit',
                         '5parent_host_ConcurrentUploadCount',
                         '5parent_host_UploadCount', '5parent_host_UploadFailedCount',

                         '5parent_host_cpu_LogicalCount', '5parent_host_cpu_PhysicalCount', '5parent_host_cpu_Percent',
                         '5parent_host_cpu_ProcessPercent',

                         '5parent_host_cpu_cputime_User', '5parent_host_cpu_cputime_System',
                         '5parent_host_cpu_cputime_Idle',
                         '5parent_host_cpu_cputime_Nice', '5parent_host_cpu_cputime_Iowait',
                         '5parent_host_cpu_cputime_Irq',
                         '5parent_host_cpu_cputime_Softirq', '5parent_host_cpu_cputime_Steal',
                         '5parent_host_cpu_cputime_Guest',
                         '5parent_host_cpu_cputime_GuestNice',

                         '5parent_host_cpu_memory_Total', '5parent_host_cpu_memory_Available',
                         '5parent_host_cpu_memory_Used',
                         '5parent_host_cpu_memory_UsedPercent', '5parent_host_cpu_memory_ProcessUsedPercent',
                         '5parent_host_cpu_memory_Free',

                         '5parent_host_cpu_network_TCPConnectionCount',
                         '5parent_host_cpu_network_UploadTCPConnectionCount',
                         '5parent_host_cpu_network_Location', '5parent_host_cpu_network_IDC',

                         '5parent_host_cpu_disk_Total', '5parent_host_cpu_disk_Free', '5parent_host_cpu_disk_Used',
                         '5parent_host_cpu_disk_UsedPercent',
                         '5parent_host_cpu_disk_InodesTotal', '5parent_host_cpu_disk_InodesUsed',
                         '5parent_host_cpu_disk_InodesFree',
                         '5parent_host_cpu_disk_InodesUsedPercent',

                         '5parent_host_cpu_build_GitVersion', '5parent_host_cpu_build_GitCommit',
                         '5parent_host_cpu_build_GoVersion',
                         '5parent_host_cpu_build_Platform',

                         '5parent_host_SchedulerClusterID', '5parent_host_CreatedAt', '5parent_host_UpdatedAt',

                         # parent 0 piece
                         '5parent_0piece_Length', '5parent_0piece_cost', '5parent_0piece_createdAt',
                         '5parent_1piece_Length', '5parent_1piece_cost', '5parent_1piece_createdAt',
                         '5parent_2piece_Length', '5parent_2piece_cost', '5parent_2piece_createdAt',
                         '5parent_3piece_Length', '5parent_3piece_cost', '5parent_3piece_createdAt',
                         '5parent_4piece_Length', '5parent_4piece_cost', '5parent_4piece_createdAt',
                         '5parent_5piece_Length', '5parent_5piece_cost', '5parent_5piece_createdAt',
                         '5parent_6piece_Length', '5parent_6piece_cost', '5parent_6piece_createdAt',
                         '5parent_7piece_Length', '5parent_7piece_cost', '5parent_7piece_createdAt',
                         '5parent_8piece_Length', '5parent_8piece_cost', '5parent_8piece_createdAt',
                         '5parent_9piece_Length', '5parent_9piece_cost', '5parent_9piece_createdAt',

                         '5parent_createdAt', '5parent_updatedAt',
                         # --------------------------------
                         # parent 6
                         '6parent_id', '6parent_Tag', '6parent_Application', '6parent_State', '6parent_Cost',
                         '6parent_UploadPieceCount', '6parent_FinishedPieceCount',

                         '6parent_host_id', '6parent_host_type', '6parent_host_hostname', '6parent_host_ip',
                         '6parent_host_port', '6parent_host_downloadPort',
                         '6parent_host_OS', '6parent_host_platform', '6parent_host_PlatformFamily',
                         '6parent_host_PlatformVersion',
                         '6parent_host_KernelVersion', '6parent_host_ConcurrentUploadLimit',
                         '6parent_host_ConcurrentUploadCount',
                         '6parent_host_UploadCount', '6parent_host_UploadFailedCount',

                         '6parent_host_cpu_LogicalCount', '6parent_host_cpu_PhysicalCount', '6parent_host_cpu_Percent',
                         '6parent_host_cpu_ProcessPercent',

                         '6parent_host_cpu_cputime_User', '6parent_host_cpu_cputime_System',
                         '6parent_host_cpu_cputime_Idle',
                         '6parent_host_cpu_cputime_Nice', '6parent_host_cpu_cputime_Iowait',
                         '6parent_host_cpu_cputime_Irq',
                         '6parent_host_cpu_cputime_Softirq', '6parent_host_cpu_cputime_Steal',
                         '6parent_host_cpu_cputime_Guest',
                         '6parent_host_cpu_cputime_GuestNice',

                         '6parent_host_cpu_memory_Total', '6parent_host_cpu_memory_Available',
                         '6parent_host_cpu_memory_Used',
                         '6parent_host_cpu_memory_UsedPercent', '6parent_host_cpu_memory_ProcessUsedPercent',
                         '6parent_host_cpu_memory_Free',

                         '6parent_host_cpu_network_TCPConnectionCount',
                         '6parent_host_cpu_network_UploadTCPConnectionCount',
                         '6parent_host_cpu_network_Location', '6parent_host_cpu_network_IDC',

                         '6parent_host_cpu_disk_Total', '6parent_host_cpu_disk_Free', '6parent_host_cpu_disk_Used',
                         '6parent_host_cpu_disk_UsedPercent',
                         '6parent_host_cpu_disk_InodesTotal', '6parent_host_cpu_disk_InodesUsed',
                         '6parent_host_cpu_disk_InodesFree',
                         '6parent_host_cpu_disk_InodesUsedPercent',

                         '6parent_host_cpu_build_GitVersion', '6parent_host_cpu_build_GitCommit',
                         '6parent_host_cpu_build_GoVersion',
                         '6parent_host_cpu_build_Platform',

                         '6parent_host_SchedulerClusterID', '6parent_host_CreatedAt', '6parent_host_UpdatedAt',

                         # parent 0 piece
                         '6parent_0piece_Length', '6parent_0piece_cost', '6parent_0piece_createdAt',
                         '6parent_1piece_Length', '6parent_1piece_cost', '6parent_1piece_createdAt',
                         '6parent_2piece_Length', '6parent_2piece_cost', '6parent_2piece_createdAt',
                         '6parent_3piece_Length', '6parent_3piece_cost', '6parent_3piece_createdAt',
                         '6parent_4piece_Length', '6parent_4piece_cost', '6parent_4piece_createdAt',
                         '6parent_5piece_Length', '6parent_5piece_cost', '6parent_5piece_createdAt',
                         '6parent_6piece_Length', '6parent_6piece_cost', '6parent_6piece_createdAt',
                         '6parent_7piece_Length', '6parent_7piece_cost', '6parent_7piece_createdAt',
                         '6parent_8piece_Length', '6parent_8piece_cost', '6parent_8piece_createdAt',
                         '6parent_9piece_Length', '6parent_9piece_cost', '6parent_9piece_createdAt',

                         '6parent_createdAt', '6parent_updatedAt',
                         # --------------------------------
                         # parent 7
                         '7parent_id', '7parent_Tag', '7parent_Application', '7parent_State', '7parent_Cost',
                         '7parent_UploadPieceCount', '7parent_FinishedPieceCount',

                         '7parent_host_id', '7parent_host_type', '7parent_host_hostname', '7parent_host_ip',
                         '7parent_host_port', '7parent_host_downloadPort',
                         '7parent_host_OS', '7parent_host_platform', '7parent_host_PlatformFamily',
                         '7parent_host_PlatformVersion',
                         '7parent_host_KernelVersion', '7parent_host_ConcurrentUploadLimit',
                         '7parent_host_ConcurrentUploadCount',
                         '7parent_host_UploadCount', '7parent_host_UploadFailedCount',

                         '7parent_host_cpu_LogicalCount', '7parent_host_cpu_PhysicalCount', '7parent_host_cpu_Percent',
                         '7parent_host_cpu_ProcessPercent',

                         '7parent_host_cpu_cputime_User', '7parent_host_cpu_cputime_System',
                         '7parent_host_cpu_cputime_Idle',
                         '7parent_host_cpu_cputime_Nice', '7parent_host_cpu_cputime_Iowait',
                         '7parent_host_cpu_cputime_Irq',
                         '7parent_host_cpu_cputime_Softirq', '7parent_host_cpu_cputime_Steal',
                         '7parent_host_cpu_cputime_Guest',
                         '7parent_host_cpu_cputime_GuestNice',

                         '7parent_host_cpu_memory_Total', '7parent_host_cpu_memory_Available',
                         '7parent_host_cpu_memory_Used',
                         '7parent_host_cpu_memory_UsedPercent', '7parent_host_cpu_memory_ProcessUsedPercent',
                         '7parent_host_cpu_memory_Free',

                         '7parent_host_cpu_network_TCPConnectionCount',
                         '7parent_host_cpu_network_UploadTCPConnectionCount',
                         '7parent_host_cpu_network_Location', '7parent_host_cpu_network_IDC',

                         '7parent_host_cpu_disk_Total', '7parent_host_cpu_disk_Free', '7parent_host_cpu_disk_Used',
                         '7parent_host_cpu_disk_UsedPercent',
                         '7parent_host_cpu_disk_InodesTotal', '7parent_host_cpu_disk_InodesUsed',
                         '7parent_host_cpu_disk_InodesFree',
                         '7parent_host_cpu_disk_InodesUsedPercent',

                         '7parent_host_cpu_build_GitVersion', '7parent_host_cpu_build_GitCommit',
                         '7parent_host_cpu_build_GoVersion',
                         '7parent_host_cpu_build_Platform',

                         '7parent_host_SchedulerClusterID', '7parent_host_CreatedAt', '7parent_host_UpdatedAt',

                         # parent 0 piece
                         '7parent_0piece_Length', '7parent_0piece_cost', '7parent_0piece_createdAt',
                         '7parent_1piece_Length', '7parent_1piece_cost', '7parent_1piece_createdAt',
                         '7parent_2piece_Length', '7parent_2piece_cost', '7parent_2piece_createdAt',
                         '7parent_3piece_Length', '7parent_3piece_cost', '7parent_3piece_createdAt',
                         '7parent_4piece_Length', '7parent_4piece_cost', '7parent_4piece_createdAt',
                         '7parent_5piece_Length', '7parent_5piece_cost', '7parent_5piece_createdAt',
                         '7parent_6piece_Length', '7parent_6piece_cost', '7parent_6piece_createdAt',
                         '7parent_7piece_Length', '7parent_7piece_cost', '7parent_7piece_createdAt',
                         '7parent_8piece_Length', '7parent_8piece_cost', '7parent_8piece_createdAt',
                         '7parent_9piece_Length', '7parent_9piece_cost', '7parent_9piece_createdAt',

                         '7parent_createdAt', '7parent_updatedAt',
                         # --------------------------------
                         # parent 8
                         '8parent_id', '8parent_Tag', '8parent_Application', '8parent_State', '8parent_Cost',
                         '8parent_UploadPieceCount', '8parent_FinishedPieceCount',

                         '8parent_host_id', '8parent_host_type', '8parent_host_hostname', '8parent_host_ip',
                         '8parent_host_port', '8parent_host_downloadPort',
                         '8parent_host_OS', '8parent_host_platform', '8parent_host_PlatformFamily',
                         '8parent_host_PlatformVersion',
                         '8parent_host_KernelVersion', '8parent_host_ConcurrentUploadLimit',
                         '8parent_host_ConcurrentUploadCount',
                         '8parent_host_UploadCount', '8parent_host_UploadFailedCount',

                         '8parent_host_cpu_LogicalCount', '8parent_host_cpu_PhysicalCount', '8parent_host_cpu_Percent',
                         '8parent_host_cpu_ProcessPercent',

                         '8parent_host_cpu_cputime_User', '8parent_host_cpu_cputime_System',
                         '8parent_host_cpu_cputime_Idle',
                         '8parent_host_cpu_cputime_Nice', '8parent_host_cpu_cputime_Iowait',
                         '8parent_host_cpu_cputime_Irq',
                         '8parent_host_cpu_cputime_Softirq', '8parent_host_cpu_cputime_Steal',
                         '8parent_host_cpu_cputime_Guest',
                         '8parent_host_cpu_cputime_GuestNice',

                         '8parent_host_cpu_memory_Total', '8parent_host_cpu_memory_Available',
                         '8parent_host_cpu_memory_Used',
                         '8parent_host_cpu_memory_UsedPercent', '8parent_host_cpu_memory_ProcessUsedPercent',
                         '8parent_host_cpu_memory_Free',

                         '8parent_host_cpu_network_TCPConnectionCount',
                         '8parent_host_cpu_network_UploadTCPConnectionCount',
                         '8parent_host_cpu_network_Location', '8parent_host_cpu_network_IDC',

                         '8parent_host_cpu_disk_Total', '8parent_host_cpu_disk_Free', '8parent_host_cpu_disk_Used',
                         '8parent_host_cpu_disk_UsedPercent',
                         '8parent_host_cpu_disk_InodesTotal', '8parent_host_cpu_disk_InodesUsed',
                         '8parent_host_cpu_disk_InodesFree',
                         '8parent_host_cpu_disk_InodesUsedPercent',

                         '8parent_host_cpu_build_GitVersion', '8parent_host_cpu_build_GitCommit',
                         '8parent_host_cpu_build_GoVersion',
                         '8parent_host_cpu_build_Platform',

                         '8parent_host_SchedulerClusterID', '8parent_host_CreatedAt', '8parent_host_UpdatedAt',

                         # parent 0 piece
                         '8parent_0piece_Length', '8parent_0piece_cost', '8parent_0piece_createdAt',
                         '8parent_1piece_Length', '8parent_1piece_cost', '8parent_1piece_createdAt',
                         '8parent_2piece_Length', '8parent_2piece_cost', '8parent_2piece_createdAt',
                         '8parent_3piece_Length', '8parent_3piece_cost', '8parent_3piece_createdAt',
                         '8parent_4piece_Length', '8parent_4piece_cost', '8parent_4piece_createdAt',
                         '8parent_5piece_Length', '8parent_5piece_cost', '8parent_5piece_createdAt',
                         '8parent_6piece_Length', '8parent_6piece_cost', '8parent_6piece_createdAt',
                         '8parent_7piece_Length', '8parent_7piece_cost', '8parent_7piece_createdAt',
                         '8parent_8piece_Length', '8parent_8piece_cost', '8parent_8piece_createdAt',
                         '8parent_9piece_Length', '8parent_9piece_cost', '8parent_9piece_createdAt',

                         '8parent_createdAt', '8parent_updatedAt',
                         # --------------------------------
                         # parent 9
                         '9parent_id', '9parent_Tag', '9parent_Application', '9parent_State', '9parent_Cost',
                         '9parent_UploadPieceCount', '9parent_FinishedPieceCount',

                         '9parent_host_id', '9parent_host_type', '9parent_host_hostname', '9parent_host_ip',
                         '9parent_host_port', '9parent_host_downloadPort',
                         '9parent_host_OS', '9parent_host_platform', '9parent_host_PlatformFamily',
                         '9parent_host_PlatformVersion',
                         '9parent_host_KernelVersion', '9parent_host_ConcurrentUploadLimit',
                         '9parent_host_ConcurrentUploadCount',
                         '9parent_host_UploadCount', '9parent_host_UploadFailedCount',

                         '9parent_host_cpu_LogicalCount', '9parent_host_cpu_PhysicalCount', '9parent_host_cpu_Percent',
                         '9parent_host_cpu_ProcessPercent',

                         '9parent_host_cpu_cputime_User', '9parent_host_cpu_cputime_System',
                         '9parent_host_cpu_cputime_Idle',
                         '9parent_host_cpu_cputime_Nice', '9parent_host_cpu_cputime_Iowait',
                         '9parent_host_cpu_cputime_Irq',
                         '9parent_host_cpu_cputime_Softirq', '9parent_host_cpu_cputime_Steal',
                         '9parent_host_cpu_cputime_Guest',
                         '9parent_host_cpu_cputime_GuestNice',

                         '9parent_host_cpu_memory_Total', '9parent_host_cpu_memory_Available',
                         '9parent_host_cpu_memory_Used',
                         '9parent_host_cpu_memory_UsedPercent', '9parent_host_cpu_memory_ProcessUsedPercent',
                         '9parent_host_cpu_memory_Free',

                         '9parent_host_cpu_network_TCPConnectionCount',
                         '9parent_host_cpu_network_UploadTCPConnectionCount',
                         '9parent_host_cpu_network_Location', '9parent_host_cpu_network_IDC',

                         '9parent_host_cpu_disk_Total', '9parent_host_cpu_disk_Free', '9parent_host_cpu_disk_Used',
                         '9parent_host_cpu_disk_UsedPercent',
                         '9parent_host_cpu_disk_InodesTotal', '9parent_host_cpu_disk_InodesUsed',
                         '9parent_host_cpu_disk_InodesFree',
                         '9parent_host_cpu_disk_InodesUsedPercent',

                         '9parent_host_cpu_build_GitVersion', '9parent_host_cpu_build_GitCommit',
                         '9parent_host_cpu_build_GoVersion',
                         '9parent_host_cpu_build_Platform',

                         '9parent_host_SchedulerClusterID', '9parent_host_CreatedAt', '9parent_host_UpdatedAt',

                         # parent 0 piece
                         '9parent_0piece_Length', '9parent_0piece_cost', '9parent_0piece_createdAt',
                         '9parent_1piece_Length', '9parent_1piece_cost', '9parent_1piece_createdAt',
                         '9parent_2piece_Length', '9parent_2piece_cost', '9parent_2piece_createdAt',
                         '9parent_3piece_Length', '9parent_3piece_cost', '9parent_3piece_createdAt',
                         '9parent_4piece_Length', '9parent_4piece_cost', '9parent_4piece_createdAt',
                         '9parent_5piece_Length', '9parent_5piece_cost', '9parent_5piece_createdAt',
                         '9parent_6piece_Length', '9parent_6piece_cost', '9parent_6piece_createdAt',
                         '9parent_7piece_Length', '9parent_7piece_cost', '9parent_7piece_createdAt',
                         '9parent_8piece_Length', '9parent_8piece_cost', '9parent_8piece_createdAt',
                         '9parent_9piece_Length', '9parent_9piece_cost', '9parent_9piece_createdAt',

                         '9parent_createdAt', '9parent_updatedAt',
                         # --------------------------------
                         # parent 10
                         '10parent_id', '10parent_Tag', '10parent_Application', '10parent_State', '10parent_Cost',
                         '10parent_UploadPieceCount', '10parent_FinishedPieceCount',

                         '10parent_host_id', '10parent_host_type', '10parent_host_hostname', '10parent_host_ip',
                         '10parent_host_port', '10parent_host_downloadPort',
                         '10parent_host_OS', '10parent_host_platform', '10parent_host_PlatformFamily',
                         '10parent_host_PlatformVersion',
                         '10parent_host_KernelVersion', '10parent_host_ConcurrentUploadLimit',
                         '10parent_host_ConcurrentUploadCount',
                         '10parent_host_UploadCount', '10parent_host_UploadFailedCount',

                         '10parent_host_cpu_LogicalCount', '10parent_host_cpu_PhysicalCount',
                         '10parent_host_cpu_Percent',
                         '10parent_host_cpu_ProcessPercent',

                         '10parent_host_cpu_cputime_User', '10parent_host_cpu_cputime_System',
                         '10parent_host_cpu_cputime_Idle',
                         '10parent_host_cpu_cputime_Nice', '10parent_host_cpu_cputime_Iowait',
                         '10parent_host_cpu_cputime_Irq',
                         '10parent_host_cpu_cputime_Softirq', '10parent_host_cpu_cputime_Steal',
                         '10parent_host_cpu_cputime_Guest',
                         '10parent_host_cpu_cputime_GuestNice',

                         '10parent_host_cpu_memory_Total', '10parent_host_cpu_memory_Available',
                         '10parent_host_cpu_memory_Used',
                         '10parent_host_cpu_memory_UsedPercent', '10parent_host_cpu_memory_ProcessUsedPercent',
                         '10parent_host_cpu_memory_Free',

                         '10parent_host_cpu_network_TCPConnectionCount',
                         '10parent_host_cpu_network_UploadTCPConnectionCount',
                         '10parent_host_cpu_network_Location', '10parent_host_cpu_network_IDC',

                         '10parent_host_cpu_disk_Total', '10parent_host_cpu_disk_Free', '10parent_host_cpu_disk_Used',
                         '10parent_host_cpu_disk_UsedPercent',
                         '10parent_host_cpu_disk_InodesTotal', '10parent_host_cpu_disk_InodesUsed',
                         '10parent_host_cpu_disk_InodesFree',
                         '10parent_host_cpu_disk_InodesUsedPercent',

                         '10parent_host_cpu_build_GitVersion', '10parent_host_cpu_build_GitCommit',
                         '10parent_host_cpu_build_GoVersion',
                         '10parent_host_cpu_build_Platform',

                         '10parent_host_SchedulerClusterID', '10parent_host_CreatedAt', '10parent_host_UpdatedAt',

                         # parent 0 piece
                         '10parent_0piece_Length', '10parent_0piece_cost', '10parent_0piece_createdAt',
                         '10parent_1piece_Length', '10parent_1piece_cost', '10parent_1piece_createdAt',
                         '10parent_2piece_Length', '10parent_2piece_cost', '10parent_2piece_createdAt',
                         '10parent_3piece_Length', '10parent_3piece_cost', '10parent_3piece_createdAt',
                         '10parent_4piece_Length', '10parent_4piece_cost', '10parent_4piece_createdAt',
                         '10parent_5piece_Length', '10parent_5piece_cost', '10parent_5piece_createdAt',
                         '10parent_6piece_Length', '10parent_6piece_cost', '10parent_6piece_createdAt',
                         '10parent_7piece_Length', '10parent_7piece_cost', '10parent_7piece_createdAt',
                         '10parent_8piece_Length', '10parent_8piece_cost', '10parent_8piece_createdAt',
                         '10parent_9piece_Length', '10parent_9piece_cost', '10parent_9piece_createdAt',

                         '10parent_createdAt', '10parent_updatedAt',
                         # --------------------------------
                         # parent 11
                         '11parent_id', '11parent_Tag', '11parent_Application', '11parent_State', '11parent_Cost',
                         '11parent_UploadPieceCount', '11parent_FinishedPieceCount',

                         '11parent_host_id', '11parent_host_type', '11parent_host_hostname', '11parent_host_ip',
                         '11parent_host_port', '11parent_host_downloadPort',
                         '11parent_host_OS', '11parent_host_platform', '11parent_host_PlatformFamily',
                         '11parent_host_PlatformVersion',
                         '11parent_host_KernelVersion', '11parent_host_ConcurrentUploadLimit',
                         '11parent_host_ConcurrentUploadCount',
                         '11parent_host_UploadCount', '11parent_host_UploadFailedCount',

                         '11parent_host_cpu_LogicalCount', '11parent_host_cpu_PhysicalCount',
                         '11parent_host_cpu_Percent',
                         '11parent_host_cpu_ProcessPercent',

                         '11parent_host_cpu_cputime_User', '11parent_host_cpu_cputime_System',
                         '11parent_host_cpu_cputime_Idle',
                         '11parent_host_cpu_cputime_Nice', '11parent_host_cpu_cputime_Iowait',
                         '11parent_host_cpu_cputime_Irq',
                         '11parent_host_cpu_cputime_Softirq', '11parent_host_cpu_cputime_Steal',
                         '11parent_host_cpu_cputime_Guest',
                         '11parent_host_cpu_cputime_GuestNice',

                         '11parent_host_cpu_memory_Total', '11parent_host_cpu_memory_Available',
                         '11parent_host_cpu_memory_Used',
                         '11parent_host_cpu_memory_UsedPercent', '11parent_host_cpu_memory_ProcessUsedPercent',
                         '11parent_host_cpu_memory_Free',

                         '11parent_host_cpu_network_TCPConnectionCount',
                         '11parent_host_cpu_network_UploadTCPConnectionCount',
                         '11parent_host_cpu_network_Location', '11parent_host_cpu_network_IDC',

                         '11parent_host_cpu_disk_Total', '11parent_host_cpu_disk_Free', '11parent_host_cpu_disk_Used',
                         '11parent_host_cpu_disk_UsedPercent',
                         '11parent_host_cpu_disk_InodesTotal', '11parent_host_cpu_disk_InodesUsed',
                         '11parent_host_cpu_disk_InodesFree',
                         '11parent_host_cpu_disk_InodesUsedPercent',

                         '11parent_host_cpu_build_GitVersion', '11parent_host_cpu_build_GitCommit',
                         '11parent_host_cpu_build_GoVersion',
                         '11parent_host_cpu_build_Platform',

                         '11parent_host_SchedulerClusterID', '11parent_host_CreatedAt', '11parent_host_UpdatedAt',

                         # parent 0 piece
                         '11parent_0piece_Length', '11parent_0piece_cost', '11parent_0piece_createdAt',
                         '11parent_1piece_Length', '11parent_1piece_cost', '11parent_1piece_createdAt',
                         '11parent_2piece_Length', '11parent_2piece_cost', '11parent_2piece_createdAt',
                         '11parent_3piece_Length', '11parent_3piece_cost', '11parent_3piece_createdAt',
                         '11parent_4piece_Length', '11parent_4piece_cost', '11parent_4piece_createdAt',
                         '11parent_5piece_Length', '11parent_5piece_cost', '11parent_5piece_createdAt',
                         '11parent_6piece_Length', '11parent_6piece_cost', '11parent_6piece_createdAt',
                         '11parent_7piece_Length', '11parent_7piece_cost', '11parent_7piece_createdAt',
                         '11parent_8piece_Length', '11parent_8piece_cost', '11parent_8piece_createdAt',
                         '11parent_9piece_Length', '11parent_9piece_cost', '11parent_9piece_createdAt',

                         '11parent_createdAt', '11parent_updatedAt',
                         # --------------------------------
                         # parent 12
                         '12parent_id', '12parent_Tag', '12parent_Application', '12parent_State', '12parent_Cost',
                         '12parent_UploadPieceCount', '12parent_FinishedPieceCount',

                         '12parent_host_id', '12parent_host_type', '12parent_host_hostname', '12parent_host_ip',
                         '12parent_host_port', '12parent_host_downloadPort',
                         '12parent_host_OS', '12parent_host_platform', '12parent_host_PlatformFamily',
                         '12parent_host_PlatformVersion',
                         '12parent_host_KernelVersion', '12parent_host_ConcurrentUploadLimit',
                         '12parent_host_ConcurrentUploadCount',
                         '12parent_host_UploadCount', '12parent_host_UploadFailedCount',

                         '12parent_host_cpu_LogicalCount', '12parent_host_cpu_PhysicalCount',
                         '12parent_host_cpu_Percent',
                         '12parent_host_cpu_ProcessPercent',

                         '12parent_host_cpu_cputime_User', '12parent_host_cpu_cputime_System',
                         '12parent_host_cpu_cputime_Idle',
                         '12parent_host_cpu_cputime_Nice', '12parent_host_cpu_cputime_Iowait',
                         '12parent_host_cpu_cputime_Irq',
                         '12parent_host_cpu_cputime_Softirq', '12parent_host_cpu_cputime_Steal',
                         '12parent_host_cpu_cputime_Guest',
                         '12parent_host_cpu_cputime_GuestNice',

                         '12parent_host_cpu_memory_Total', '12parent_host_cpu_memory_Available',
                         '12parent_host_cpu_memory_Used',
                         '12parent_host_cpu_memory_UsedPercent', '12parent_host_cpu_memory_ProcessUsedPercent',
                         '12parent_host_cpu_memory_Free',

                         '12parent_host_cpu_network_TCPConnectionCount',
                         '12parent_host_cpu_network_UploadTCPConnectionCount',
                         '12parent_host_cpu_network_Location', '12parent_host_cpu_network_IDC',

                         '12parent_host_cpu_disk_Total', '12parent_host_cpu_disk_Free', '12parent_host_cpu_disk_Used',
                         '12parent_host_cpu_disk_UsedPercent',
                         '12parent_host_cpu_disk_InodesTotal', '12parent_host_cpu_disk_InodesUsed',
                         '12parent_host_cpu_disk_InodesFree',
                         '12parent_host_cpu_disk_InodesUsedPercent',

                         '12parent_host_cpu_build_GitVersion', '12parent_host_cpu_build_GitCommit',
                         '12parent_host_cpu_build_GoVersion',
                         '12parent_host_cpu_build_Platform',

                         '12parent_host_SchedulerClusterID', '12parent_host_CreatedAt', '12parent_host_UpdatedAt',

                         # parent 0 piece
                         '12parent_0piece_Length', '12parent_0piece_cost', '12parent_0piece_createdAt',
                         '12parent_1piece_Length', '12parent_1piece_cost', '12parent_1piece_createdAt',
                         '12parent_2piece_Length', '12parent_2piece_cost', '12parent_2piece_createdAt',
                         '12parent_3piece_Length', '12parent_3piece_cost', '12parent_3piece_createdAt',
                         '12parent_4piece_Length', '12parent_4piece_cost', '12parent_4piece_createdAt',
                         '12parent_5piece_Length', '12parent_5piece_cost', '12parent_5piece_createdAt',
                         '12parent_6piece_Length', '12parent_6piece_cost', '12parent_6piece_createdAt',
                         '12parent_7piece_Length', '12parent_7piece_cost', '12parent_7piece_createdAt',
                         '12parent_8piece_Length', '12parent_8piece_cost', '12parent_8piece_createdAt',
                         '12parent_9piece_Length', '12parent_9piece_cost', '12parent_9piece_createdAt',

                         '12parent_createdAt', '12parent_updatedAt',
                         # --------------------------------
                         # parent 13
                         '13parent_id', '13parent_Tag', '13parent_Application', '13parent_State', '13parent_Cost',
                         '13parent_UploadPieceCount', '13parent_FinishedPieceCount',

                         '13parent_host_id', '13parent_host_type', '13parent_host_hostname', '13parent_host_ip',
                         '13parent_host_port', '13parent_host_downloadPort',
                         '13parent_host_OS', '13parent_host_platform', '13parent_host_PlatformFamily',
                         '13parent_host_PlatformVersion',
                         '13parent_host_KernelVersion', '13parent_host_ConcurrentUploadLimit',
                         '13parent_host_ConcurrentUploadCount',
                         '13parent_host_UploadCount', '13parent_host_UploadFailedCount',

                         '13parent_host_cpu_LogicalCount', '13parent_host_cpu_PhysicalCount',
                         '13parent_host_cpu_Percent',
                         '13parent_host_cpu_ProcessPercent',

                         '13parent_host_cpu_cputime_User', '13parent_host_cpu_cputime_System',
                         '13parent_host_cpu_cputime_Idle',
                         '13parent_host_cpu_cputime_Nice', '13parent_host_cpu_cputime_Iowait',
                         '13parent_host_cpu_cputime_Irq',
                         '13parent_host_cpu_cputime_Softirq', '13parent_host_cpu_cputime_Steal',
                         '13parent_host_cpu_cputime_Guest',
                         '13parent_host_cpu_cputime_GuestNice',

                         '13parent_host_cpu_memory_Total', '13parent_host_cpu_memory_Available',
                         '13parent_host_cpu_memory_Used',
                         '13parent_host_cpu_memory_UsedPercent', '13parent_host_cpu_memory_ProcessUsedPercent',
                         '13parent_host_cpu_memory_Free',

                         '13parent_host_cpu_network_TCPConnectionCount',
                         '13parent_host_cpu_network_UploadTCPConnectionCount',
                         '13parent_host_cpu_network_Location', '13parent_host_cpu_network_IDC',

                         '13parent_host_cpu_disk_Total', '13parent_host_cpu_disk_Free', '13parent_host_cpu_disk_Used',
                         '13parent_host_cpu_disk_UsedPercent',
                         '13parent_host_cpu_disk_InodesTotal', '13parent_host_cpu_disk_InodesUsed',
                         '13parent_host_cpu_disk_InodesFree',
                         '13parent_host_cpu_disk_InodesUsedPercent',

                         '13parent_host_cpu_build_GitVersion', '13parent_host_cpu_build_GitCommit',
                         '13parent_host_cpu_build_GoVersion',
                         '13parent_host_cpu_build_Platform',

                         '13parent_host_SchedulerClusterID', '13parent_host_CreatedAt', '13parent_host_UpdatedAt',

                         # parent 0 piece
                         '13parent_0piece_Length', '13parent_0piece_cost', '13parent_0piece_createdAt',
                         '13parent_1piece_Length', '13parent_1piece_cost', '13parent_1piece_createdAt',
                         '13parent_2piece_Length', '13parent_2piece_cost', '13parent_2piece_createdAt',
                         '13parent_3piece_Length', '13parent_3piece_cost', '13parent_3piece_createdAt',
                         '13parent_4piece_Length', '13parent_4piece_cost', '13parent_4piece_createdAt',
                         '13parent_5piece_Length', '13parent_5piece_cost', '13parent_5piece_createdAt',
                         '13parent_6piece_Length', '13parent_6piece_cost', '13parent_6piece_createdAt',
                         '13parent_7piece_Length', '13parent_7piece_cost', '13parent_7piece_createdAt',
                         '13parent_8piece_Length', '13parent_8piece_cost', '13parent_8piece_createdAt',
                         '13parent_9piece_Length', '13parent_9piece_cost', '13parent_9piece_createdAt',

                         '13parent_createdAt', '13parent_updatedAt',
                         # --------------------------------
                         # parent 14
                         '14parent_id', '14parent_Tag', '14parent_Application', '14parent_State', '14parent_Cost',
                         '14parent_UploadPieceCount', '14parent_FinishedPieceCount',

                         '14parent_host_id', '14parent_host_type', '14parent_host_hostname', '14parent_host_ip',
                         '14parent_host_port', '14parent_host_downloadPort',
                         '14parent_host_OS', '14parent_host_platform', '14parent_host_PlatformFamily',
                         '14parent_host_PlatformVersion',
                         '14parent_host_KernelVersion', '14parent_host_ConcurrentUploadLimit',
                         '14parent_host_ConcurrentUploadCount',
                         '14parent_host_UploadCount', '14parent_host_UploadFailedCount',

                         '14parent_host_cpu_LogicalCount', '14parent_host_cpu_PhysicalCount',
                         '14parent_host_cpu_Percent',
                         '14parent_host_cpu_ProcessPercent',

                         '14parent_host_cpu_cputime_User', '14parent_host_cpu_cputime_System',
                         '14parent_host_cpu_cputime_Idle',
                         '14parent_host_cpu_cputime_Nice', '14parent_host_cpu_cputime_Iowait',
                         '14parent_host_cpu_cputime_Irq',
                         '14parent_host_cpu_cputime_Softirq', '14parent_host_cpu_cputime_Steal',
                         '14parent_host_cpu_cputime_Guest',
                         '14parent_host_cpu_cputime_GuestNice',

                         '14parent_host_cpu_memory_Total', '14parent_host_cpu_memory_Available',
                         '14parent_host_cpu_memory_Used',
                         '14parent_host_cpu_memory_UsedPercent', '14parent_host_cpu_memory_ProcessUsedPercent',
                         '14parent_host_cpu_memory_Free',

                         '14parent_host_cpu_network_TCPConnectionCount',
                         '14parent_host_cpu_network_UploadTCPConnectionCount',
                         '14parent_host_cpu_network_Location', '14parent_host_cpu_network_IDC',

                         '14parent_host_cpu_disk_Total', '14parent_host_cpu_disk_Free', '14parent_host_cpu_disk_Used',
                         '14parent_host_cpu_disk_UsedPercent',
                         '14parent_host_cpu_disk_InodesTotal', '14parent_host_cpu_disk_InodesUsed',
                         '14parent_host_cpu_disk_InodesFree',
                         '14parent_host_cpu_disk_InodesUsedPercent',

                         '14parent_host_cpu_build_GitVersion', '14parent_host_cpu_build_GitCommit',
                         '14parent_host_cpu_build_GoVersion',
                         '14parent_host_cpu_build_Platform',

                         '14parent_host_SchedulerClusterID', '14parent_host_CreatedAt', '14parent_host_UpdatedAt',

                         # parent 0 piece
                         '14parent_0piece_Length', '14parent_0piece_cost', '14parent_0piece_createdAt',
                         '14parent_1piece_Length', '14parent_1piece_cost', '14parent_1piece_createdAt',
                         '14parent_2piece_Length', '14parent_2piece_cost', '14parent_2piece_createdAt',
                         '14parent_3piece_Length', '14parent_3piece_cost', '14parent_3piece_createdAt',
                         '14parent_4piece_Length', '14parent_4piece_cost', '14parent_4piece_createdAt',
                         '14parent_5piece_Length', '14parent_5piece_cost', '14parent_5piece_createdAt',
                         '14parent_6piece_Length', '14parent_6piece_cost', '14parent_6piece_createdAt',
                         '14parent_7piece_Length', '14parent_7piece_cost', '14parent_7piece_createdAt',
                         '14parent_8piece_Length', '14parent_8piece_cost', '14parent_8piece_createdAt',
                         '14parent_9piece_Length', '14parent_9piece_cost', '14parent_9piece_createdAt',

                         '14parent_createdAt', '14parent_updatedAt',
                         # --------------------------------
                         # parent 15
                         '15parent_id', '15parent_Tag', '15parent_Application', '15parent_State', '15parent_Cost',
                         '15parent_UploadPieceCount', '15parent_FinishedPieceCount',

                         '15parent_host_id', '15parent_host_type', '15parent_host_hostname', '15parent_host_ip',
                         '15parent_host_port', '15parent_host_downloadPort',
                         '15parent_host_OS', '15parent_host_platform', '15parent_host_PlatformFamily',
                         '15parent_host_PlatformVersion',
                         '15parent_host_KernelVersion', '15parent_host_ConcurrentUploadLimit',
                         '15parent_host_ConcurrentUploadCount',
                         '15parent_host_UploadCount', '15parent_host_UploadFailedCount',

                         '15parent_host_cpu_LogicalCount', '15parent_host_cpu_PhysicalCount',
                         '15parent_host_cpu_Percent',
                         '15parent_host_cpu_ProcessPercent',

                         '15parent_host_cpu_cputime_User', '15parent_host_cpu_cputime_System',
                         '15parent_host_cpu_cputime_Idle',
                         '15parent_host_cpu_cputime_Nice', '15parent_host_cpu_cputime_Iowait',
                         '15parent_host_cpu_cputime_Irq',
                         '15parent_host_cpu_cputime_Softirq', '15parent_host_cpu_cputime_Steal',
                         '15parent_host_cpu_cputime_Guest',
                         '15parent_host_cpu_cputime_GuestNice',

                         '15parent_host_cpu_memory_Total', '15parent_host_cpu_memory_Available',
                         '15parent_host_cpu_memory_Used',
                         '15parent_host_cpu_memory_UsedPercent', '15parent_host_cpu_memory_ProcessUsedPercent',
                         '15parent_host_cpu_memory_Free',

                         '15parent_host_cpu_network_TCPConnectionCount',
                         '15parent_host_cpu_network_UploadTCPConnectionCount',
                         '15parent_host_cpu_network_Location', '15parent_host_cpu_network_IDC',

                         '15parent_host_cpu_disk_Total', '15parent_host_cpu_disk_Free', '15parent_host_cpu_disk_Used',
                         '15parent_host_cpu_disk_UsedPercent',
                         '15parent_host_cpu_disk_InodesTotal', '15parent_host_cpu_disk_InodesUsed',
                         '15parent_host_cpu_disk_InodesFree',
                         '15parent_host_cpu_disk_InodesUsedPercent',

                         '15parent_host_cpu_build_GitVersion', '15parent_host_cpu_build_GitCommit',
                         '15parent_host_cpu_build_GoVersion',
                         '15parent_host_cpu_build_Platform',

                         '15parent_host_SchedulerClusterID', '15parent_host_CreatedAt', '15parent_host_UpdatedAt',

                         # parent 0 piece
                         '15parent_0piece_Length', '15parent_0piece_cost', '15parent_0piece_createdAt',
                         '15parent_1piece_Length', '15parent_1piece_cost', '15parent_1piece_createdAt',
                         '15parent_2piece_Length', '15parent_2piece_cost', '15parent_2piece_createdAt',
                         '15parent_3piece_Length', '15parent_3piece_cost', '15parent_3piece_createdAt',
                         '15parent_4piece_Length', '15parent_4piece_cost', '15parent_4piece_createdAt',
                         '15parent_5piece_Length', '15parent_5piece_cost', '15parent_5piece_createdAt',
                         '15parent_6piece_Length', '15parent_6piece_cost', '15parent_6piece_createdAt',
                         '15parent_7piece_Length', '15parent_7piece_cost', '15parent_7piece_createdAt',
                         '15parent_8piece_Length', '15parent_8piece_cost', '15parent_8piece_createdAt',
                         '15parent_9piece_Length', '15parent_9piece_cost', '15parent_9piece_createdAt',

                         '15parent_createdAt', '15parent_updatedAt',
                         # --------------------------------
                         # parent 16
                         '16parent_id', '16parent_Tag', '16parent_Application', '16parent_State', '16parent_Cost',
                         '16parent_UploadPieceCount', '16parent_FinishedPieceCount',

                         '16parent_host_id', '16parent_host_type', '16parent_host_hostname', '16parent_host_ip',
                         '16parent_host_port', '16parent_host_downloadPort',
                         '16parent_host_OS', '16parent_host_platform', '16parent_host_PlatformFamily',
                         '16parent_host_PlatformVersion',
                         '16parent_host_KernelVersion', '16parent_host_ConcurrentUploadLimit',
                         '16parent_host_ConcurrentUploadCount',
                         '16parent_host_UploadCount', '16parent_host_UploadFailedCount',

                         '16parent_host_cpu_LogicalCount', '16parent_host_cpu_PhysicalCount',
                         '16parent_host_cpu_Percent',
                         '16parent_host_cpu_ProcessPercent',

                         '16parent_host_cpu_cputime_User', '16parent_host_cpu_cputime_System',
                         '16parent_host_cpu_cputime_Idle',
                         '16parent_host_cpu_cputime_Nice', '16parent_host_cpu_cputime_Iowait',
                         '16parent_host_cpu_cputime_Irq',
                         '16parent_host_cpu_cputime_Softirq', '16parent_host_cpu_cputime_Steal',
                         '16parent_host_cpu_cputime_Guest',
                         '16parent_host_cpu_cputime_GuestNice',

                         '16parent_host_cpu_memory_Total', '16parent_host_cpu_memory_Available',
                         '16parent_host_cpu_memory_Used',
                         '16parent_host_cpu_memory_UsedPercent', '16parent_host_cpu_memory_ProcessUsedPercent',
                         '16parent_host_cpu_memory_Free',

                         '16parent_host_cpu_network_TCPConnectionCount',
                         '16parent_host_cpu_network_UploadTCPConnectionCount',
                         '16parent_host_cpu_network_Location', '16parent_host_cpu_network_IDC',

                         '16parent_host_cpu_disk_Total', '16parent_host_cpu_disk_Free', '16parent_host_cpu_disk_Used',
                         '16parent_host_cpu_disk_UsedPercent',
                         '16parent_host_cpu_disk_InodesTotal', '16parent_host_cpu_disk_InodesUsed',
                         '16parent_host_cpu_disk_InodesFree',
                         '16parent_host_cpu_disk_InodesUsedPercent',

                         '16parent_host_cpu_build_GitVersion', '16parent_host_cpu_build_GitCommit',
                         '16parent_host_cpu_build_GoVersion',
                         '16parent_host_cpu_build_Platform',

                         '16parent_host_SchedulerClusterID', '16parent_host_CreatedAt', '16parent_host_UpdatedAt',

                         # parent 0 piece
                         '16parent_0piece_Length', '16parent_0piece_cost', '16parent_0piece_createdAt',
                         '16parent_1piece_Length', '16parent_1piece_cost', '16parent_1piece_createdAt',
                         '16parent_2piece_Length', '16parent_2piece_cost', '16parent_2piece_createdAt',
                         '16parent_3piece_Length', '16parent_3piece_cost', '16parent_3piece_createdAt',
                         '16parent_4piece_Length', '16parent_4piece_cost', '16parent_4piece_createdAt',
                         '16parent_5piece_Length', '16parent_5piece_cost', '16parent_5piece_createdAt',
                         '16parent_6piece_Length', '16parent_6piece_cost', '16parent_6piece_createdAt',
                         '16parent_7piece_Length', '16parent_7piece_cost', '16parent_7piece_createdAt',
                         '16parent_8piece_Length', '16parent_8piece_cost', '16parent_8piece_createdAt',
                         '16parent_9piece_Length', '16parent_9piece_cost', '16parent_9piece_createdAt',

                         '16parent_createdAt', '16parent_updatedAt',
                         # --------------------------------
                         # parent 17
                         '17parent_id', '17parent_Tag', '17parent_Application', '17parent_State', '17parent_Cost',
                         '17parent_UploadPieceCount', '17parent_FinishedPieceCount',

                         '17parent_host_id', '17parent_host_type', '17parent_host_hostname', '17parent_host_ip',
                         '17parent_host_port', '17parent_host_downloadPort',
                         '17parent_host_OS', '17parent_host_platform', '17parent_host_PlatformFamily',
                         '17parent_host_PlatformVersion',
                         '17parent_host_KernelVersion', '17parent_host_ConcurrentUploadLimit',
                         '17parent_host_ConcurrentUploadCount',
                         '17parent_host_UploadCount', '17parent_host_UploadFailedCount',

                         '17parent_host_cpu_LogicalCount', '17parent_host_cpu_PhysicalCount',
                         '17parent_host_cpu_Percent',
                         '17parent_host_cpu_ProcessPercent',

                         '17parent_host_cpu_cputime_User', '17parent_host_cpu_cputime_System',
                         '17parent_host_cpu_cputime_Idle',
                         '17parent_host_cpu_cputime_Nice', '17parent_host_cpu_cputime_Iowait',
                         '17parent_host_cpu_cputime_Irq',
                         '17parent_host_cpu_cputime_Softirq', '17parent_host_cpu_cputime_Steal',
                         '17parent_host_cpu_cputime_Guest',
                         '17parent_host_cpu_cputime_GuestNice',

                         '17parent_host_cpu_memory_Total', '17parent_host_cpu_memory_Available',
                         '17parent_host_cpu_memory_Used',
                         '17parent_host_cpu_memory_UsedPercent', '17parent_host_cpu_memory_ProcessUsedPercent',
                         '17parent_host_cpu_memory_Free',

                         '17parent_host_cpu_network_TCPConnectionCount',
                         '17parent_host_cpu_network_UploadTCPConnectionCount',
                         '17parent_host_cpu_network_Location', '17parent_host_cpu_network_IDC',

                         '17parent_host_cpu_disk_Total', '17parent_host_cpu_disk_Free', '17parent_host_cpu_disk_Used',
                         '17parent_host_cpu_disk_UsedPercent',
                         '17parent_host_cpu_disk_InodesTotal', '17parent_host_cpu_disk_InodesUsed',
                         '17parent_host_cpu_disk_InodesFree',
                         '17parent_host_cpu_disk_InodesUsedPercent',

                         '17parent_host_cpu_build_GitVersion', '17parent_host_cpu_build_GitCommit',
                         '17parent_host_cpu_build_GoVersion',
                         '17parent_host_cpu_build_Platform',

                         '17parent_host_SchedulerClusterID', '17parent_host_CreatedAt', '17parent_host_UpdatedAt',

                         # parent 0 piece
                         '17parent_0piece_Length', '17parent_0piece_cost', '17parent_0piece_createdAt',
                         '17parent_1piece_Length', '17parent_1piece_cost', '17parent_1piece_createdAt',
                         '17parent_2piece_Length', '17parent_2piece_cost', '17parent_2piece_createdAt',
                         '17parent_3piece_Length', '17parent_3piece_cost', '17parent_3piece_createdAt',
                         '17parent_4piece_Length', '17parent_4piece_cost', '17parent_4piece_createdAt',
                         '17parent_5piece_Length', '17parent_5piece_cost', '17parent_5piece_createdAt',
                         '17parent_6piece_Length', '17parent_6piece_cost', '17parent_6piece_createdAt',
                         '17parent_7piece_Length', '17parent_7piece_cost', '17parent_7piece_createdAt',
                         '17parent_8piece_Length', '17parent_8piece_cost', '17parent_8piece_createdAt',
                         '17parent_9piece_Length', '17parent_9piece_cost', '17parent_9piece_createdAt',

                         '17parent_createdAt', '17parent_updatedAt',
                         # --------------------------------
                         # parent 18
                         '18parent_id', '18parent_Tag', '18parent_Application', '18parent_State', '18parent_Cost',
                         '18parent_UploadPieceCount', '18parent_FinishedPieceCount',

                         '18parent_host_id', '18parent_host_type', '18parent_host_hostname', '18parent_host_ip',
                         '18parent_host_port', '18parent_host_downloadPort',
                         '18parent_host_OS', '18parent_host_platform', '18parent_host_PlatformFamily',
                         '18parent_host_PlatformVersion',
                         '18parent_host_KernelVersion', '18parent_host_ConcurrentUploadLimit',
                         '18parent_host_ConcurrentUploadCount',
                         '18parent_host_UploadCount', '18parent_host_UploadFailedCount',

                         '18parent_host_cpu_LogicalCount', '18parent_host_cpu_PhysicalCount',
                         '18parent_host_cpu_Percent',
                         '18parent_host_cpu_ProcessPercent',

                         '18parent_host_cpu_cputime_User', '18parent_host_cpu_cputime_System',
                         '18parent_host_cpu_cputime_Idle',
                         '18parent_host_cpu_cputime_Nice', '18parent_host_cpu_cputime_Iowait',
                         '18parent_host_cpu_cputime_Irq',
                         '18parent_host_cpu_cputime_Softirq', '18parent_host_cpu_cputime_Steal',
                         '18parent_host_cpu_cputime_Guest',
                         '18parent_host_cpu_cputime_GuestNice',

                         '18parent_host_cpu_memory_Total', '18parent_host_cpu_memory_Available',
                         '18parent_host_cpu_memory_Used',
                         '18parent_host_cpu_memory_UsedPercent', '18parent_host_cpu_memory_ProcessUsedPercent',
                         '18parent_host_cpu_memory_Free',

                         '18parent_host_cpu_network_TCPConnectionCount',
                         '18parent_host_cpu_network_UploadTCPConnectionCount',
                         '18parent_host_cpu_network_Location', '18parent_host_cpu_network_IDC',

                         '18parent_host_cpu_disk_Total', '18parent_host_cpu_disk_Free', '18parent_host_cpu_disk_Used',
                         '18parent_host_cpu_disk_UsedPercent',
                         '18parent_host_cpu_disk_InodesTotal', '18parent_host_cpu_disk_InodesUsed',
                         '18parent_host_cpu_disk_InodesFree',
                         '18parent_host_cpu_disk_InodesUsedPercent',

                         '18parent_host_cpu_build_GitVersion', '18parent_host_cpu_build_GitCommit',
                         '18parent_host_cpu_build_GoVersion',
                         '18parent_host_cpu_build_Platform',

                         '18parent_host_SchedulerClusterID', '18parent_host_CreatedAt', '18parent_host_UpdatedAt',

                         # parent 0 piece
                         '18parent_0piece_Length', '18parent_0piece_cost', '18parent_0piece_createdAt',
                         '18parent_1piece_Length', '18parent_1piece_cost', '18parent_1piece_createdAt',
                         '18parent_2piece_Length', '18parent_2piece_cost', '18parent_2piece_createdAt',
                         '18parent_3piece_Length', '18parent_3piece_cost', '18parent_3piece_createdAt',
                         '18parent_4piece_Length', '18parent_4piece_cost', '18parent_4piece_createdAt',
                         '18parent_5piece_Length', '18parent_5piece_cost', '18parent_5piece_createdAt',
                         '18parent_6piece_Length', '18parent_6piece_cost', '18parent_6piece_createdAt',
                         '18parent_7piece_Length', '18parent_7piece_cost', '18parent_7piece_createdAt',
                         '18parent_8piece_Length', '18parent_8piece_cost', '18parent_8piece_createdAt',
                         '18parent_9piece_Length', '18parent_9piece_cost', '18parent_9piece_createdAt',

                         '18parent_createdAt', '18parent_updatedAt',
                         # --------------------------------
                         # parent 19
                         '19parent_id', '19parent_Tag', '19parent_Application', '19parent_State', '19parent_Cost',
                         '19parent_UploadPieceCount', '19parent_FinishedPieceCount',

                         '19parent_host_id', '19parent_host_type', '19parent_host_hostname', '19parent_host_ip',
                         '19parent_host_port', '19parent_host_downloadPort',
                         '19parent_host_OS', '19parent_host_platform', '19parent_host_PlatformFamily',
                         '19parent_host_PlatformVersion',
                         '19parent_host_KernelVersion', '19parent_host_ConcurrentUploadLimit',
                         '19parent_host_ConcurrentUploadCount',
                         '19parent_host_UploadCount', '19parent_host_UploadFailedCount',

                         '19parent_host_cpu_LogicalCount', '19parent_host_cpu_PhysicalCount',
                         '19parent_host_cpu_Percent',
                         '19parent_host_cpu_ProcessPercent',

                         '19parent_host_cpu_cputime_User', '19parent_host_cpu_cputime_System',
                         '19parent_host_cpu_cputime_Idle',
                         '19parent_host_cpu_cputime_Nice', '19parent_host_cpu_cputime_Iowait',
                         '19parent_host_cpu_cputime_Irq',
                         '19parent_host_cpu_cputime_Softirq', '19parent_host_cpu_cputime_Steal',
                         '19parent_host_cpu_cputime_Guest',
                         '19parent_host_cpu_cputime_GuestNice',

                         '19parent_host_cpu_memory_Total', '19parent_host_cpu_memory_Available',
                         '19parent_host_cpu_memory_Used',
                         '19parent_host_cpu_memory_UsedPercent', '19parent_host_cpu_memory_ProcessUsedPercent',
                         '19parent_host_cpu_memory_Free',

                         '19parent_host_cpu_network_TCPConnectionCount',
                         '19parent_host_cpu_network_UploadTCPConnectionCount',
                         '19parent_host_cpu_network_Location', '19parent_host_cpu_network_IDC',

                         '19parent_host_cpu_disk_Total', '19parent_host_cpu_disk_Free', '19parent_host_cpu_disk_Used',
                         '19parent_host_cpu_disk_UsedPercent',
                         '19parent_host_cpu_disk_InodesTotal', '19parent_host_cpu_disk_InodesUsed',
                         '19parent_host_cpu_disk_InodesFree',
                         '19parent_host_cpu_disk_InodesUsedPercent',

                         '19parent_host_cpu_build_GitVersion', '19parent_host_cpu_build_GitCommit',
                         '19parent_host_cpu_build_GoVersion',
                         '19parent_host_cpu_build_Platform',

                         '19parent_host_SchedulerClusterID', '19parent_host_CreatedAt', '19parent_host_UpdatedAt',

                         # parent 0 piece
                         '19parent_0piece_Length', '19parent_0piece_cost', '19parent_0piece_createdAt',
                         '19parent_1piece_Length', '19parent_1piece_cost', '19parent_1piece_createdAt',
                         '19parent_2piece_Length', '19parent_2piece_cost', '19parent_2piece_createdAt',
                         '19parent_3piece_Length', '19parent_3piece_cost', '19parent_3piece_createdAt',
                         '19parent_4piece_Length', '19parent_4piece_cost', '19parent_4piece_createdAt',
                         '19parent_5piece_Length', '19parent_5piece_cost', '19parent_5piece_createdAt',
                         '19parent_6piece_Length', '19parent_6piece_cost', '19parent_6piece_createdAt',
                         '19parent_7piece_Length', '19parent_7piece_cost', '19parent_7piece_createdAt',
                         '19parent_8piece_Length', '19parent_8piece_cost', '19parent_8piece_createdAt',
                         '19parent_9piece_Length', '19parent_9piece_cost', '19parent_9piece_createdAt',

                         '19parent_createdAt', '19parent_updatedAt',
                         # --------------------------------
                         'createdAt', 'updatedAt',
                     ])
    print(df)

    bandwidth_map = {}
    # 遍历csv, 将host_id 转成index
    for index in df.index:
        src_id = df.loc[index]['host_id']
        print(src_id)
        if node_index.get(src_id) is None:
            continue

        for i in range(20):
            parent_key = str(i) + 'parent_id'
            print(parent_key)
            dest_id = df.loc[index][parent_key]

            print(dest_id)
            if dest_id != '':
                for j in range(10):
                    pieceLength_key = parent_key + '_' + str(j) + 'piece_Length'
                    pieceLength = df.loc[index][pieceLength_key]
                    pieceCost_key = parent_key + '_' + str(j) + 'piece_cost'
                    pieceCost = df.loc[index][pieceCost_key]
                    if pieceLength != '':
                        bandwidth = float(pieceLength) / float(pieceCost)

                        # 存起来
                        src_index = node_index[src_id]
                        dest_index = node_index[dest_id]

                        bandwidth_key = src_index + ':' + dest_index

                        if bandwidth_map.get(bandwidth_key) is None:
                            bandwidth_map[bandwidth_key] = bandwidth
                        else:
                            if bandwidth_map[bandwidth_key] < bandwidth:
                                bandwidth_map[bandwidth_key] = bandwidth

    print(bandwidth_map)
    return bandwidth_map

if __name__ == '__main__':
    node_num, feature, adj_lists, node_index = read_nt()
    # 把host id 和index 映射关系传一下，方便后面使用
    # bandwidth_map = read_d(node_index)

    #
    # # 暂时先拟定IDC 是10维，然后location  2| 3 | 3 | = 2*3*3 共 18维，然后IP 是32维
    #
    # idc_dim = 10
    # location_dim = 2 * 3 * 3
    # ip_dim = 32
    # input_dim = idc_dim + location_dim + ip_dim
    #
    # INTERNAL_DIM = 128
    # SAMPLE_SIZES = [5, 5]
    # LEARNING_RATE = 0.001
    #
    # graphsage = GraphSage(input_dim, INTERNAL_DIM, LEARNING_RATE)
    # graphsage.train()
    #
    # print(graphsage.summary())
    # tf.saved_model.save(
    #     graphsage,
    #     "keras/graphsage",
    #     signatures={
    #         "call": graphsage.call,
    #         "train": graphsage.train,
    #     },
    # )
