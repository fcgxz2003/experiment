import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
from preprocess import read_nt
from minibatch import build_batch_from_nodes


def client_init(url="210.30.96.107:30883",
                ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False,
                verbose=False):
    if ssl:
        ssl_options = {}
        if key_file is not None:
            ssl_options['keyfile'] = key_file
        if cert_file is not None:
            ssl_options['certfile'] = cert_file
        if ca_certs is not None:
            ssl_options['ca_certs'] = ca_certs
        ssl_context_factory = None
        if insecure:
            ssl_context_factory = gevent.ssl._create_unverified_context
        triton_client = httpclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=True,
            ssl_options=ssl_options,
            insecure=insecure,
            ssl_context_factory=ssl_context_factory)
    else:
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=verbose)

    return triton_client


def infer(triton_client):
    SAMPLE_SIZES = [5, 5]
    node_num, feature, adj_lists, node_index = read_nt()
    src_nodes0, dstsrc2srcs0, dstsrc2dsts0, dif_mats0 = build_batch_from_nodes([1], adj_lists, SAMPLE_SIZES)
    src_nodes1, dstsrc2srcs1, dstsrc2dsts1, dif_mats1 = build_batch_from_nodes([2], adj_lists, SAMPLE_SIZES)

    inputs = []
    outputs = []

    # batch_size=8
    # 如果batch_size超过配置文件的max_batch_size，infer则会报错
    inputs.append(httpclient.InferInput('dif_mat0_1', dif_mats0[0].shape, "FP32"))
    inputs.append(httpclient.InferInput('dif_mat0_2', dif_mats0[1].shape, "FP32"))
    inputs.append(httpclient.InferInput('dif_mat1_1', dif_mats1[0].shape, "FP32"))
    inputs.append(httpclient.InferInput('dif_mat0_2', dif_mats1[1].shape, "FP32"))

    inputs.append(httpclient.InferInput('dstsrc2dst0_1', dstsrc2dsts0[0].shape, "INT64"))
    inputs.append(httpclient.InferInput('dstsrc2dst0_2', dstsrc2dsts0[1].shape, "INT64"))
    inputs.append(httpclient.InferInput('dstsrc2dst1_1', dstsrc2dsts1[0].shape, "INT64"))
    inputs.append(httpclient.InferInput('dstsrc2dst1_2', dstsrc2dsts1[1].shape, "INT64"))

    inputs.append(httpclient.InferInput('dstsrc2src0_1', dstsrc2srcs0[0].shape, "INT64"))
    inputs.append(httpclient.InferInput('dstsrc2src0_2', dstsrc2srcs0[1].shape, "INT64"))
    inputs.append(httpclient.InferInput('dstsrc2src1_1', dstsrc2srcs1[0].shape, "INT64"))
    inputs.append(httpclient.InferInput('dstsrc2src1_2', dstsrc2srcs1[1].shape, "INT64"))

    inputs.append(httpclient.InferInput('src_nodes0', src_nodes0.shape, "INT32"))
    inputs.append(httpclient.InferInput('src_nodes1', src_nodes1.shape, "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(dif_mats0[0].astype(np.float32), binary_data=False)
    inputs[1].set_data_from_numpy(dif_mats0[1].astype(np.float32), binary_data=False)
    inputs[2].set_data_from_numpy(dif_mats1[0].astype(np.float32), binary_data=False)
    inputs[3].set_data_from_numpy(dif_mats1[1].astype(np.float32), binary_data=False)

    inputs[4].set_data_from_numpy(dstsrc2dsts0[0].astype(np.int64), binary_data=False)
    inputs[5].set_data_from_numpy(dstsrc2dsts0[1].astype(np.int64), binary_data=False)
    inputs[6].set_data_from_numpy(dstsrc2dsts1[0].astype(np.int64), binary_data=False)
    inputs[7].set_data_from_numpy(dstsrc2dsts1[1].astype(np.int64), binary_data=False)

    inputs[8].set_data_from_numpy(dstsrc2srcs0[0].astype(np.int64), binary_data=False)
    inputs[9].set_data_from_numpy(dstsrc2srcs0[1].astype(np.int64), binary_data=False)
    inputs[10].set_data_from_numpy(dstsrc2srcs1[0].astype(np.int64), binary_data=False)
    inputs[11].set_data_from_numpy(dstsrc2srcs1[1].astype(np.int64), binary_data=False)

    inputs[12].set_data_from_numpy(src_nodes0.astype(np.int32), binary_data=False)
    inputs[13].set_data_from_numpy(src_nodes1.astype(np.int32), binary_data=False)

    # OUTPUT0为配置文件中的输出节点名称
    outputs.append(httpclient.InferRequestedOutput('output_0', binary_data=False))

    results = triton_client.infer(
        model_name='tf_savemodel_without_train',
        model_version='1',
        inputs=inputs,
        outputs=outputs,
    )

    print(results.get_response())


if __name__ == '__main__':
    # c = client_init()
    c = httpclient.InferenceServerClient(url='210.30.96.107:30883')
    infer(c)
