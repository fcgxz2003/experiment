import tensorflow as tf
from tensorflow.keras import layers


class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        """
        :param ndarray((#(node), #(feature))) features: a matrix, each row is feature for a node
        """
        super().__init__(trainable=False, **kwargs)
        self.features = tf.constant(features)

    def call(self, nodes):
        """
        :param [int] nodes: node ids
        """
        return tf.gather(self.features, nodes)


class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, activ=True, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)
        self.activ_fn = tf.nn.relu if activ else tf.identity
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
        dst_features = tf.gather(dstsrc_features, dstsrc2dst)
        src_features = tf.gather(dstsrc_features, dstsrc2src)
        aggregated_features = tf.matmul(dif_mat, src_features)
        concatenated_features = tf.concat([aggregated_features, dst_features], 1)
        x = tf.matmul(concatenated_features, self.w)
        return self.activ_fn(x)



class graphsage(tf.keras.Model):
    def __init__(self, raw_features, internal_dim, neg_weight, learning_rate):
        """
        结合了 Graphsage 和 MLP 的 Q value table.
        :param raw_features:
        :param internal_dim:
        :param neg_weight:
        :param learning_rate:
        """
        # GraphSage
        num_layers = 2  # 这里默认是两层采样，后面需要修改
        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'

        super().__init__()

        self.input_layer = RawFeature(raw_features, name="raw_feature_layer")
        self.agg_ly1 = MeanAggregator(raw_features.shape[-1]
                                      , internal_dim
                                      , name="agg_lv1"
                                      , activ=False
                                      )

        self.agg_ly2 = MeanAggregator(internal_dim
                                      , internal_dim
                                      , name="agg_lv2"
                                      , activ=True
                                      )
        self.neg_weight = neg_weight

        # MLP
        # TODO：维度的修正，输出多少个维度？用几层网络。
        self.dense_ly1 = layers.Dense(64, activation='relu', dtype='float64')
        self.dense_ly2 = layers.Dense(64, activation='relu', dtype='float64')
        self.dense_ly3 = layers.Dense(8, dtype='float64')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._loss = tf.keras.losses.compute_uloss

    def call(self, minibatch):
        # GraphSage
        x = self.input_layer.call(tf.squeeze(minibatch.src_nodes))
        x = self.agg_ly1.call(x
                              , minibatch.dstsrc2srcs.pop()
                              , minibatch.dstsrc2dsts.pop()
                              , minibatch.dif_mats.pop()
                              )

        x = self.agg_ly1.call(x
                              , minibatch.dstsrc2srcs.pop()
                              , minibatch.dstsrc2dsts.pop()
                              , minibatch.dif_mats.pop()
                              )

        # MLP
        embeddingABN = tf.math.l2_normalize(x, 1)
        x = self.dense_ly1.call(embeddingABN)
        x = self.dense_ly2.call(x)
        x = self.dense_ly3.call(x)

        return x

    def train(self, minibatch):
        with tf.GradientTape() as tape:
            choose_pro = self.call(minibatch)
            self._loss = (self.compute_uloss(choose_pro))
            loss = self._loss
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss

        # TODO 需要魔改损失函数

    def compute_uloss(self, embeddingA, embeddingB, embeddingN, neg_weight):
        """
                compute and return the loss for unspervised model based on Eq (1) in the
                GraphSage paper

                :param 2d-tensor embeddingA: embedding of a list of nodes
                :param 2d-tensor embeddingB: embedding of a list of neighbor nodes
                                             pairwise to embeddingA
                :param 2d-tensor embeddingN: embedding of a list of non-neighbor nodes
                                             (negative samples) to embeddingA
                :param float neg_weight: negative weight
                """

        # positive affinity: pair-wise calculation
        pos_affinity = tf.reduce_sum(tf.multiply(embeddingA, embeddingB), axis=1)
        # negative affinity: enumeration of all combinations of (embeddingA, embeddingN)
        neg_affinity = tf.matmul(embeddingA, tf.transpose(embeddingN))

        pos_xent = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(pos_affinity)
                                                           , pos_affinity
                                                           , "positive_xent")
        neg_xent = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(neg_affinity)
                                                           , neg_affinity
                                                           , "negative_xent")

        weighted_neg = tf.multiply(neg_weight, tf.reduce_sum(neg_xent))
        batch_loss = tf.add(tf.reduce_sum(pos_xent), weighted_neg)

        # per batch loss: GraphSAGE:models.py line 378
        return tf.divide(batch_loss, embeddingA.shape[0])