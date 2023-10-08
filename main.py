import tensorflow as tf
from preprocess import read_nt
from minibatch import build_batch_from_nodes
from graphsage import GraphSage

if __name__ == "__main__":
    # 暂时先拟定IDC 是10维，然后location  4| 6 | 8 |  共 18维，然后IP 是32维

    idc_dim = 10
    location_dim = 2 * 3 * 3
    ip_dim = 32

    INTERNAL_DIM = 60
    SAMPLE_SIZES = [3, 2]
    LEARNING_RATE = 0.001

    node_num, feature, adj_lists, node_index = read_nt()
    # print('-------')
    # print(src_nodes)
    # print(dstsrc2srcs)
    # print(dstsrc2dsts)
    # print(dif_mats)

    print(type(feature))

    # 瞎弄几个带宽
    pieceLength = [[15728640 * 10]]
    pieceCost = [[52905000]]

    pieceLength = tf.constant(pieceLength, dtype=tf.float64)
    pieceCost = tf.constant(pieceCost, dtype=tf.float64)

    graphsage = GraphSage(feature, INTERNAL_DIM, len(SAMPLE_SIZES))

    src_nodes0, dstsrc2srcs0, dstsrc2dsts0, dif_mats0 = build_batch_from_nodes([0,1], adj_lists, SAMPLE_SIZES)
    src_nodes1, dstsrc2srcs1, dstsrc2dsts1, dif_mats1 = build_batch_from_nodes([1,2], adj_lists, SAMPLE_SIZES)

    print(src_nodes0)
    print(dstsrc2srcs0[0])
    print(dstsrc2srcs0[1])
    print(dstsrc2dsts0[0])
    print(dstsrc2dsts0[1])
    print(dif_mats0[0])
    print(dif_mats0[1])

    print(src_nodes1)
    print(dstsrc2srcs1[0])
    print(dstsrc2srcs1[1])
    print(dstsrc2dsts1[0])
    print(dstsrc2dsts1[1])
    print(dif_mats1[0])
    print(dif_mats1[1])

    loss = graphsage.train(tf.constant(src_nodes0),
                           tf.constant(dstsrc2srcs0[0]), tf.constant(dstsrc2srcs0[1]),
                           tf.constant(dstsrc2dsts0[0]), tf.constant(dstsrc2dsts0[1]),
                           tf.constant(dif_mats0[0]), tf.constant(dif_mats0[1]),

                           tf.constant(src_nodes1),
                           tf.constant(dstsrc2srcs1[0]), tf.constant(dstsrc2srcs1[1]),
                           tf.constant(dstsrc2dsts1[0]), tf.constant(dstsrc2dsts1[1]),
                           tf.constant(dif_mats1[0]), tf.constant(dif_mats1[1]),

                           tf.divide(pieceLength, pieceCost))

    print("loss:", loss)

    src_nodes0, dstsrc2srcs0, dstsrc2dsts0, dif_mats0 = build_batch_from_nodes([1], adj_lists, SAMPLE_SIZES)
    src_nodes1, dstsrc2srcs1, dstsrc2dsts1, dif_mats1 = build_batch_from_nodes([2], adj_lists, SAMPLE_SIZES)

    predicted_value = graphsage.call(tf.constant(src_nodes0),
                                     tf.constant(dstsrc2srcs0[0]), tf.constant(dstsrc2srcs0[1]),
                                     tf.constant(dstsrc2dsts0[0]), tf.constant(dstsrc2dsts0[1]),
                                     tf.constant(dif_mats0[0]), tf.constant(dif_mats0[1]),

                                     tf.constant(src_nodes1),
                                     tf.constant(dstsrc2srcs1[0]), tf.constant(dstsrc2srcs1[1]),
                                     tf.constant(dstsrc2dsts1[0]), tf.constant(dstsrc2dsts1[1]),
                                     tf.constant(dif_mats1[0]), tf.constant(dif_mats1[1]))
    print("predicted_value:", predicted_value)

    # # print(graphsage.summary())
    # tf.saved_model.save(
    #     graphsage,
    #     "graphsage",
    #     signatures={
    #         "call": graphsage.call,
    #         "train": graphsage.train,
    #     },
    # )
