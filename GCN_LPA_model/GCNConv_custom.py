import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_sparse import SparseTensor, matmul

class GCNConv_custom(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_custom, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weights):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)



        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index,edge_weight=edge_weights, x=x)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x):
        return matmul(adj_t, x, reduce=self.aggr)
