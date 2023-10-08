import redis
import tensorflow as tf
from graphsage import GraphSage

if __name__ == '__main__':
    rdb = redis.Redis(host='210.30.96.102', port=30205, db=3, password='dragonfly')

    INTERNAL_DIM = 60
    SAMPLE_SIZES = [5, 5]

    value = rdb.lrange("feature", 0, -1)
    feature = [eval(i) for i in value]
    graphsage = GraphSage(feature, INTERNAL_DIM, len(SAMPLE_SIZES))

    # 先走一遍把模型保存下来
    # print(graphsage.summary())
    tf.saved_model.save(
        graphsage,
        "graphsage",
        signatures={
            "call": graphsage.call,
            "train": graphsage.train,
        },
    )