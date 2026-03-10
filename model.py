import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_classes=2, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)
        return x