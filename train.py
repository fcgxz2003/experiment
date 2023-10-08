import redis
import tensorflow as tf
from graphsage import GraphSage
import collections

if __name__ == '__main__':
    rdb = redis.Redis(host='210.30.96.102', port=30205, db=3, password='dragonfly')

    graphsage = tf.saved_model.load("graphsage")

    # 执行多久save 新模型
    save_new_model_times = 32
    times = 0

    while True:
        if times >= save_new_model_times:
            print("save new model")

            tf.saved_model.save(
                graphsage,
                "graphsage",
                signatures={
                    "call": graphsage.call,
                    "train": graphsage.train,
                },
            )
            times = 0

        keys = rdb.keys('train:*')
        for key in keys:
            hash_flag = rdb.hget(key, "flag")
            true = rdb.hget(key, "true")
            if true is not None and hash_flag is not None and 0 < eval(true) < 200:
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

                loss = graphsage.train(tf.constant(srcNodes1),
                                       tf.constant(dstsrc2srcs1[0]), tf.constant(dstsrc2srcs1[1]),
                                       tf.constant(dstsrc2dsts1[0]), tf.constant(dstsrc2dsts1[1]),
                                       tf.constant(difMatrix1[0]), tf.constant(difMatrix1[1]),

                                       tf.constant(srcNodes2),
                                       tf.constant(dstsrc2srcs2[0]), tf.constant(dstsrc2srcs2[1]),
                                       tf.constant(dstsrc2dsts2[0]), tf.constant(dstsrc2dsts2[1]),
                                       tf.constant(difMatrix2[0]), tf.constant(difMatrix2[1]),
                                       tf.constant([eval(true)]))

                rdb.delete(key)
                print("loss:", loss)
                times = times + 1
