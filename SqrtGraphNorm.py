import paddle
import paddle.nn as nn
import pgl


class GraphNorm(nn.Layer):
    """Implementation of graph normalization. Each node features is divied by sqrt(num_nodes) per graphs.
    
    Args:
        graph: the graph object from (:code:`Graph`)
        feature: A tensor with shape (num_nodes, feature_size).

    Return:
        A tensor with shape (num_nodes, hidden_size)

    References:

    [1] BENCHMARKING GRAPH NEURAL NETWORKS. https://arxiv.org/abs/2003.00982

    """

    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = pgl.nn.GraphPool(pool_type="sum")

    def forward(self, graph, feature):
        """graph norm"""
        nodes = paddle.ones(shape=[graph.num_nodes, 1], dtype="float32")
        norm = self.graph_pool(graph, nodes)
        norm = paddle.sqrt(norm)
        norm = paddle.gather(norm, graph.graph_node_id)
        return feature / norm
