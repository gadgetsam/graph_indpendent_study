import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

from GCN_LPA_model.GCNConv_custom import GCNConv_custom
from GCN_LPA_model.LPA_custom import LPA_custom

dataset = Planetoid(root='tmp/Cora', name='Cora')
class Net(torch.nn.Module):
    def __init__(self, num_edges):
        super(Net, self).__init__()
        self.conv1 = GCNConv_custom(dataset.num_node_features, 32)
        self.conv2 = GCNConv_custom(32, 16)
        self.conv3 = GCNConv_custom(16, 16)
        self.conv4 = GCNConv_custom(16, dataset.num_classes)
        self.weights_layer = torch.nn.Linear(num_edges, num_edges)
        self.lpa = LPA_custom(5)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)
        edge_weight = self.weights_layer(edge_weight)
        num_nodes = maybe_num_nodes(edge_index, None)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg.masked_fill_(deg == float('inf'), 0)
        edge_weight= deg[row]*edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index, edge_weight)
        x_2=self.lpa(data.y, edge_index, data.train_mask, edge_weight)
        return F.log_softmax(x, dim=1), F.log_softmax(x_2, dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_edges=13264).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out, out_2 = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])+10*F.nll_loss(out_2[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    print(epoch)
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))