if __name__ == '__main__':
    src_node1 = [0, 1, 2, 3, 32, 33]
    dstsrc2srcs0_0 = [1, 2]
    dstsrc2srcs0_1 = [0, 1, 2, 3, 4, 5]
    dstsrc2dsts0_0 = [0]
    dstsrc2dsts0_1 = [0, 1, 2]
    dif_mats0_0 = [[0.5, 0.5]]
    dif_mats0_1 = [[0., 0.33333334, 0.33333334, 0.33333334, 0., 0., ],
                   [0.33333334, 0., 0.33333334, 0., 0.33333334, 0., ],
                   [0.33333334, 0.33333334, 0., 0., 0., 0.33333334]]

    src_node2 = [0, 1, 2, 3, 12, 14, 32]
    dstsrc2srcs1_0 = [0, 2]
    dstsrc2srcs1_1 = [0, 1, 2, 3, 4, 5, 6]
    dstsrc2dsts2_0 = [1]
    dstsrc2dsts2_1 = [0, 1, 6]
    dif_mats1_0 = [[0.5, 0.5]]
    dif_mats1_1 = [[0., 0.33333334, 0.33333334, 0.33333334, 0., 0., 0., ],
                   [0.33333334, 0., 0.33333334, 0., 0., 0., 0.33333334, ],
                   [0., 0.33333334, 0., 0., 0.33333334, 0.33333334, 0., ]]

    src_node = list(set(src_node1).union(set(src_node2)))
    src_node.sort()
    print(src_node) # [0, 1, 2, 3, 12, 14, 32, 33]

    dstsrc2srcs_1 = [i for i in range(len(src_node))] # [0, 1, 2, 3, 4, 5, 6, 7]
    print(dstsrc2srcs_1)

    dstsrc2dsts_0 = dstsrc2dsts0_0 + dstsrc2dsts2_0 # [0, 1]
    print(dstsrc2dsts_0)

    dstsrc2dsts_1 = list(set(dstsrc2dsts0_1).union(set(dstsrc2dsts2_1))) # [0, 1, 2, 6]
    dstsrc2dsts_1.sort()
    print(dstsrc2dsts_1)





