#!/usr/bin/env python3

from tensorflow.keras.layers import Dense
import tensorflow as tf

init_fn = tf.keras.initializers.GlorotUniform


class GraphSage(tf.keras.Model):
    def __init__(self, raw_features, internal_dim, num_layers):
        assert num_layers > 0, 'illegal parameter "num_layers"'
        assert internal_dim > 0, 'illegal parameter "internal_dim"'
        super().__init__()
        self.input_layer = RawFeature(raw_features, name="raw_feature_layer")

        self.seq_layers = []
        for i in range(1, num_layers + 1):
            layer_name = "agg_lv" + str(i)
            input_dim = internal_dim if i > 1 else raw_features.shape[-1]
            has_activ = False if i == num_layers else True
            aggregator_layer = MeanAggregator(input_dim, internal_dim, name=layer_name, activ=has_activ)
            self.seq_layers.append(aggregator_layer)

        self.dense1 = Dense(128, activation=tf.nn.relu, name="dense1")
        self.dense1.build(input_shape=(None, internal_dim * 2))
        self.dense2 = Dense(64, activation=tf.nn.relu, name="dense2")
        self.dense2.build(input_shape=(None, 128))
        self.dense3 = Dense(32, activation=tf.nn.relu, name="dense3")
        self.dense3.build(input_shape=(None, 64))
        self.dense4 = Dense(8, activation=tf.nn.relu, name="dense4")
        self.dense4.build(input_shape=(None, 32))
        self.dense = Dense(1, name="dense")
        self.dense.build(input_shape=(None, 8))

        LEARNING_RATE = 0.5
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),

        ])
    def call(self,
             src_nodes0, dstsrc2src0_1, dstsrc2src0_2, dstsrc2dst0_1, dstsrc2dst0_2,
             dif_mat0_1, dif_mat0_2,
             src_nodes1, dstsrc2src1_1, dstsrc2src1_2, dstsrc2dst1_1, dstsrc2dst1_2,
             dif_mat1_1, dif_mat1_2):
        x = self.input_layer(tf.squeeze(src_nodes0))
        x = self.seq_layers[0](x, dstsrc2src0_2, dstsrc2dst0_2, dif_mat0_2)
        x = self.seq_layers[1](x, dstsrc2src0_1, dstsrc2dst0_1, dif_mat0_1)

        y = self.input_layer(tf.squeeze(src_nodes1))
        y = self.seq_layers[0](y, dstsrc2src1_2, dstsrc2dst1_2, dif_mat1_2)
        y = self.seq_layers[1](y, dstsrc2src1_1, dstsrc2dst1_1, dif_mat1_1)

        z = tf.concat([x, y], 1)
        z = self.dense1(z)
        z = self.dense2(z)
        z = self.dense3(z)
        z = self.dense4(z)
        return self.dense(z)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.int64),
            tf.TensorSpec(shape=(None, ), dtype=tf.int64),
        ])
    def train(self, src_nodes0, dstsrc2src0_1, dstsrc2src0_2, dstsrc2dst0_1, dstsrc2dst0_2, dif_mat0_1, dif_mat0_2,
              src_nodes1, dstsrc2src1_1, dstsrc2src1_2, dstsrc2dst1_1, dstsrc2dst1_2, dif_mat1_1, dif_mat1_2,
              piece_length, piece_cost):
        with tf.GradientTape() as tape:
            predict_value = self.call(src_nodes0, dstsrc2src0_1, dstsrc2src0_2, dstsrc2dst0_1, dstsrc2dst0_2,
                                      dif_mat0_1, dif_mat0_2,
                                      src_nodes1, dstsrc2src1_1, dstsrc2src1_2, dstsrc2dst1_1, dstsrc2dst1_2,
                                      dif_mat1_1, dif_mat1_2)
            loss = self.compute_uloss(predict_value, piece_length, piece_cost)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss

    def compute_uloss(self, predict_value, piece_length, piece_cost):
        """
        loss function 需要魔改，这三者维度相同
        :param predict_value: 预测值
        :param piece_length: 下载piece 的大小
        :param piece_cost: 下载piece 的开销
        :return: 平均损失函数
        """
        return tf.reduce_mean(tf.subtract(tf.divide(piece_length, piece_cost), tf.cast(predict_value, tf.float64)))


################################################################
#                         Custom Layers                        #
################################################################

class RawFeature(tf.keras.layers.Layer):
    def __init__(self, features, **kwargs):
        """
        :param ndarray((#(node), #(feature))) features: a matrix, each row is feature for a node
        """
        super().__init__(trainable=False, **kwargs)
        self.features = tf.constant(features, dtype=tf.float32)

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
                                 , initializer=init_fn
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
