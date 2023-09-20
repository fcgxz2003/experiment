import tensorflow as tf


class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)
        # 激活函数的选择
        self.activ_fn = tf.nn.relu if activ else tf.identity
        # 创建可训练的权重变量
        # shape = (src_dim*2, dst_dim)：指定变量的形状，即权重矩阵的维度。
        # init_fn = tf.keras.initializers.GlorotUniform 指定变量的初始化方法为 Glorot均匀分布（GlorotUniform）。
        # trainable = True：指定变量是否可训练，这里设置为 True，表示权重变量 self.w 是需要在训练过程中更新的可训练变量。
        self.w = self.add_weight(name=kwargs["name"] + "_weight"
                                 , shape=(src_dim * 2, dst_dim)
                                 , dtype=tf.float32
                                 , initializer=tf.keras.initializers.GlorotUniform
                                 , trainable=True
                                 )

    def call(self, dstsrc_features, dstsrc2src, dstsrc2dst, dif_mat):
        """
        :param tensor dstsrc_features: the embedding from the previous layer
        :param tensor dstsrc2dst: 1d index mapping (prepraed by minibatch generator)
        :param tensor dstsrc2src: 1d index mapping (prepraed by minibatch generator)
        :param tensor dif_mat: 2d diffusion matrix (prepraed by minibatch generator)
        """

        # 从dstsrc_features特征向量中，收集dstsrc2dst下标的内容，作为dst_features
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        # 从dstsrc_features特征向量中，收集dstsrc2src下标的内容，作为src_features
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        # 矩阵乘，乘上均值聚合矩阵。
        aggregated_features = tf.matmul(dif_mat, src_features)
        # 拼接aggregated_features 和 dst_features
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        # 矩阵乘，乘上权重。
        x = tf.matmul(concatenated_features, self.w)
        # self.activ_fn = tf.nn.relu if activ else tf.identity 最后走一个激活函数
        return self.activ_fn(x)


class GraphSage(tf.keras.Model):
    def __init__(self, input_dim, internal_dim, learning_rate):
        """
        GraphSage Initialization.
        :param input_dim: 输入的维数
        :param internal_dim: 中间层的维数
        :param learning_rate: 学习率
        """
        # GraphSage
        assert input_dim > 0, 'illegal parameter "input_dim"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'
        assert learning_rate > 0, 'illegal parameter "learning_rate"'

        super().__init__()
        self.agg_ly1 = MeanAggregator(input_dim
                                      , internal_dim
                                      , name="agg_lv1"
                                      , activ=True
                                      )

        self.agg_ly2 = MeanAggregator(internal_dim
                                      , internal_dim
                                      , name="agg_lv2"
                                      , activ=True
                                      )

        # MLP
        # # TODO：维度的修正，输出多少个维度？用几层网络。
        #
        # self.dense_ly1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, dtype='float64')(tf.keras.layers.Input(shape=(None,internal_dim)))
        # self.dense_ly2 = tf.keras.layers.Dense(32, activation=tf.nn.relu, dtype='float64')(tf.keras.layers.Input(shape=(None,64)))
        # self.dense_ly3 = tf.keras.layers.Dense(16, activation=tf.nn.relu, dtype='float64')(tf.keras.layers.Input(shape=(None,32)))
        # self.dense_ly4 = tf.keras.layers.Dense(8, activation=tf.nn.relu, dtype='float64')(tf.keras.layers.Input(shape=(None,16)))
        # # TODO： 是否用softmax 函数作为输出，因为是要映射到0~1之间？
        # self.dense_ly5 = tf.keras.layers.Dense(1, activation=tf.nn.softmax, dtype='float64')(tf.keras.layers.Input(shape=(None,8)))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        ])
    def call(self, src_nodes, dstsrc2src_1, dstsrc2src_2, dstsrc2dst_1, dstsrc2dst_2, dif_mat_1, dif_mat_2):
        """
        前向传播
        :param src_nodes: 这里省去了 tf.gather(self.features, nodes)这一步，直接将src结点的特征传进来即可。
        :param dstsrc2src_1:
        :param dstsrc2src_2:
        :param dstsrc2dst_1:
        :param dstsrc2dst_2:
        :param dif_mat_1:
        :param dif_mat_2:
        :return:
        """
        # GraphSage
        # 先聚合第二层，再聚合第一层
        x = self.agg_ly1.call(src_nodes, dstsrc2src_2, dstsrc2dst_2, dif_mat_2)
        x = self.agg_ly2.call(x, dstsrc2src_1, dstsrc2dst_1, dif_mat_1)

        # # MLP
        # embeddingABN = tf.math.l2_normalize(x, 1)
        # x = self.dense_ly1(embeddingABN)
        # x = self.dense_ly2(x)
        # x = self.dense_ly3(x)
        # x = self.dense_ly4(x)
        # x = self.dense_ly5(x)
        return x

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=None, dtype=tf.float32),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=None, dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        ])
    def train(self, src_nodes, dstsrc2src_1, dstsrc2src_2, dstsrc2dst_1, dstsrc2dst_2, dif_mat_1, dif_mat_2, real_value,
              piece_count):
        """
        训练过程，这里要限定好batch_size，控制好维度
        :param src_nodes: 这里省去了 tf.gather(self.features, nodes)这一步，直接将src结点的特征传进来即可。
        :param dstsrc2src_1:
        :param dstsrc2src_2:
        :param dstsrc2dst_1:
        :param dstsrc2dst_2:
        :param dif_mat_1:
        :param dif_mat_2:
        :param real_value:
        :param piece_count:
        :return: loss
        """
        with tf.GradientTape() as tape:
            predict_value = self(src_nodes, dstsrc2src_1, dstsrc2src_2, dstsrc2dst_1, dstsrc2dst_2, dif_mat_1,
                                      dif_mat_2)
            loss = self.compute_uloss(predict_value, real_value, piece_count)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss

    def compute_uloss(self, predict_value, real_value, piece_count):
        """
        loss function 需要魔改，这三者维度相同
        :param predict_value:
        :param real_value:
        :param piece_count:
        :return:
        """

        return tf.constant(2.111381)

    # def compute_uloss(self, embeddingA, embeddingB, embeddingN, neg_weight):
    #
    #     # positive affinity: pair-wise calculation
    #     pos_affinity = tf.reduce_sum(tf.multiply(embeddingA, embeddingB), axis=1)
    #     # negative affinity: enumeration of all combinations of (embeddingA, embeddingN)
    #     neg_affinity = tf.matmul(embeddingA, tf.transpose(embeddingN))
    #
    #     pos_xent = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(pos_affinity)
    #                                                        , pos_affinity
    #                                                        , "positive_xent")
    #     neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(neg_affinity)
    #                                                        , neg_affinity
    #                                                        , "negative_xent")
    #
    #     weighted_neg = tf.multiply(neg_weight, tf.reduce_sum(neg_xent))
    #     batch_loss = tf.add(tf.reduce_sum(pos_xent), weighted_neg)
    #
    #     # per batch loss: GraphSAGE:models.py line 378
    #     return tf.divide(batch_loss, embeddingA.shape[0])


if __name__ == "__main__":
    # 暂时先拟定IDC 是10维，然后location  2| 3 | 3 | = 2*3*3 共 18维，然后IP 是32维

    idc_dim = 10
    location_dim = 2 * 3 * 3
    ip_dim = 32
    input_dim = idc_dim + location_dim + ip_dim

    INTERNAL_DIM = 128
    SAMPLE_SIZES = [5, 5]
    LEARNING_RATE = 0.001

    graphsage = GraphSage(input_dim, INTERNAL_DIM, LEARNING_RATE)
    graphsage.train()

    print(graphsage.summary())
    tf.saved_model.save(
        graphsage,
        "keras/graphsage",
        signatures={
            "call": graphsage.call,
            "train": graphsage.train,
        },
    )
