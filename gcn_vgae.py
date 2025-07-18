import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

# ------------------ GCN Encoder for VGAE ------------------ #
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)

        self.residual = nn.Linear(in_channels, hidden_channels)

        self.mu = nn.Linear(hidden_channels, hidden_channels)
        self.logvar = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        identity = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.relu(self.bn4(self.conv4(x, edge_index)) + identity)
        return self.mu(x), self.logvar(x)

# ------------------ VGAE-GCN Model ------------------ #
class ResVGAE_GCN(nn.Module):
    def __init__(self, num_features_xd=78, num_features_xt=25, n_filters=32, embed_dim=128, output_dim=128, dropout=0.15):
        super(ResVGAE_GCN, self).__init__()

        self.output_dim = output_dim
        self.encoder = GCNEncoder(num_features_xd, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(output_dim)

        # Cell Line CNN
        self.conv_xt_1 = nn.Conv1d(1, n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)

        self.bn_xt_1 = nn.BatchNorm1d(n_filters)
        self.bn_xt_2 = nn.BatchNorm1d(n_filters * 2)
        self.bn_xt_3 = nn.BatchNorm1d(n_filters * 4)
        self.fc1_xt = nn.Linear(2944, output_dim)

        # Fully Connected Layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target[:, None, :]

        # VGAE Encoder
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logvar)

        # 💡 SKIPPING dense adj reconstruction to prevent OOM
        adj_reconstructed = None

        # Global pooling
        z_pooled = global_add_pool(z, batch)
        z_pooled = self.layer_norm(z_pooled)

        # CNN for cell line
        conv_xt = F.relu(self.bn_xt_1(self.conv_xt_1(target)))
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = F.relu(self.bn_xt_2(self.conv_xt_2(conv_xt)))
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = F.relu(self.bn_xt_3(self.conv_xt_3(conv_xt)))
        conv_xt = self.pool_xt_3(conv_xt)

        xt = conv_xt.view(conv_xt.shape[0], -1)
        xt = self.fc1_xt(xt)

        # Fusion and Prediction
        xc = torch.cat((z_pooled, xt), dim=1)
        xc = self.relu(self.bn_fc1(self.fc1(xc)))
        xc = self.dropout(xc)
        xc = self.relu(self.bn_fc2(self.fc2(xc)))
        xc = self.dropout(xc)
        xc = self.relu(self.bn_fc3(self.fc3(xc)))
        xc = self.dropout(xc)
        out = self.out(xc)

        return out, z_pooled
