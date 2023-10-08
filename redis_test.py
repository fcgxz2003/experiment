import redis
import tensorflow as tf
from graphsage import GraphSage
import collections
from multiprocessing import Process

dataset = {}
BatchSize = 2

if __name__ == '__main__':
    rdb = redis.Redis(host='210.30.96.102', port=30205, db=3, password='dragonfly')

    INTERNAL_DIM = 60
    SAMPLE_SIZES = [5, 5]

    value = rdb.lrange("feature", 0, -1)
    feature = [eval(i) for i in value]
    graphsage = GraphSage(feature, INTERNAL_DIM, len(SAMPLE_SIZES))

    while True:
        keys = rdb.keys('train:*')
        for key in keys:
            flag = rdb.hget(key, "flag")
            if flag is not None:
                if int(flag) == 1:
                    srcNodes1 = rdb.hget(key, "srcNodes1")
                    srcNodes1 = eval(srcNodes1)
                    dstsrc2srcs1 = rdb.hget(key, "dstsrc2srcs1")
                    dstsrc2srcs1 = eval(dstsrc2srcs1)
                    dstsrc2dsts1 = rdb.hget(key, "dstsrc2dsts1")
                    dstsrc2dsts1 = eval(dstsrc2dsts1)
                    difMatrix1 = rdb.hget(key, "difMatrix1")
                    difMatrix1 = eval(difMatrix1)

                    srcNodes2 = rdb.hget(key, "srcNodes2")
                    srcNodes2 = eval(srcNodes2)
                    dstsrc2srcs2 = rdb.hget(key, "dstsrc2srcs2")
                    dstsrc2srcs2 = eval(dstsrc2srcs2)
                    dstsrc2dsts2 = rdb.hget(key, "dstsrc2dsts2")
                    dstsrc2dsts2 = eval(dstsrc2dsts2)
                    difMatrix2 = rdb.hget(key, "difMatrix2")
                    difMatrix2 = eval(difMatrix2)

                    predicted = graphsage.call(tf.constant(srcNodes1),
                                               tf.constant(dstsrc2srcs1[0]), tf.constant(dstsrc2srcs1[1]),
                                               tf.constant(dstsrc2dsts1[0]), tf.constant(dstsrc2dsts1[1]),
                                               tf.constant(difMatrix1[0]), tf.constant(difMatrix1[1]),

                                               tf.constant(srcNodes2),
                                               tf.constant(dstsrc2srcs2[0]), tf.constant(dstsrc2srcs2[1]),
                                               tf.constant(dstsrc2dsts2[0]), tf.constant(dstsrc2dsts2[1]),
                                               tf.constant(difMatrix2[0]), tf.constant(difMatrix2[1]))

                    p = predicted.numpy()

                    rdb.hset(key, "predicted", str(p[0][0]))
                    rdb.hset(key, "flag", 0)
                    print(str(p[0][0]))

                    MiniBatchFields = ["srcNodes1", "dstsrc2srcs1_0", "dstsrc2srcs1_1", "dstsrc2dsts1_0",
                                       "dstsrc2dsts1_1",
                                       "difMatrix1_0", "difMatrix1_1",
                                       "srcNodes2", "dstsrc2srcs2_0", "dstsrc2srcs2_1", "dstsrc2dsts2_0",
                                       "dstsrc2dsts2_1",
                                       "difMatrix2_0", "difMatrix2_1",
                                       "predicted"]
                    MiniBatch = collections.namedtuple("MiniBatch", MiniBatchFields)

                    minibatch = MiniBatch(srcNodes1, dstsrc2srcs1[0], dstsrc2srcs1[1], dstsrc2dsts1[0], dstsrc2dsts1[1],
                                          difMatrix1[0], difMatrix1[1],
                                          srcNodes2, dstsrc2srcs2[0], dstsrc2srcs2[1], dstsrc2dsts2[0], dstsrc2dsts2[1],
                                          difMatrix2[0], difMatrix2[1],
                                          p[0][0])

                    dataset[key] = minibatch

            # 判断是有有要训练的内容
            hash_flag = rdb.hget(key, "flag")
            true = rdb.hget(key, "true")
            if true is not None and hash_flag is not None and 20 < eval(true) < 80:
                if dataset.get(key) is not None:
                    loss = graphsage.train(tf.constant(dataset[key].srcNodes1),
                                           tf.constant(dataset[key].dstsrc2srcs1_0),
                                           tf.constant(dataset[key].dstsrc2srcs1_1),
                                           tf.constant(dataset[key].dstsrc2dsts1_0),
                                           tf.constant(dataset[key].dstsrc2dsts1_1),
                                           tf.constant(dataset[key].difMatrix1_0),
                                           tf.constant(dataset[key].difMatrix1_1),

                                           tf.constant(dataset[key].srcNodes2),
                                           tf.constant(dataset[key].dstsrc2srcs2_0),
                                           tf.constant(dataset[key].dstsrc2srcs2_1),
                                           tf.constant(dataset[key].dstsrc2dsts2_0),
                                           tf.constant(dataset[key].dstsrc2dsts2_1),
                                           tf.constant(dataset[key].difMatrix2_0),
                                           tf.constant(dataset[key].difMatrix2_1),
                                           tf.constant(eval(true)))

                    rdb.delete(key)
                    dataset.pop(key)
                    print("loss:", loss)
