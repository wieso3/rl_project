
import torch
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lin = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin(x))
        return x


class AttKernel(nn.Module):

    def __init__(self, n_agents, n_heads, hidden_dim):
        super(AttKernel, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.att_1 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.att_2 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(self, x, mask, n_agents=None):

        n_agents = n_agents if n_agents else self.n_agents

        feature_broadcast = torch.broadcast_to(x, (n_agents, self.n_agents, self.hidden_dim))
        x_feats = torch.bmm(mask, feature_broadcast)

        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        x_feats, _ = self.att_1(x, x_feats, x_feats)
        x_feats, attn_weights = self.att_2(x, x_feats, x_feats)

        return x_feats.squeeze(1)


class QNet(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(QNet, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q


class MeanKernel(nn.Module):
    def __init__(self, hidden_dim, n_agents):
        super(MeanKernel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

    def forward(self, x, mask, n_agents=None):
        n_agents = n_agents if n_agents else self.n_agents

        feature_broadcast = torch.broadcast_to(x, (n_agents, self.n_agents, self.hidden_dim))
        x_feats = torch.bmm(mask, feature_broadcast)
        x_mean = torch.mean(x_feats, dim=1)

        return x_mean


class DGN(nn.Module):
    def __init__(self, n_agents, input_dim, hidden_dim, n_actions, use_att=True):
        super(DGN, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim)

        if use_att:
            self.att_1 = AttKernel(n_agents, 4, hidden_dim)
        else:
            self.att_1 = MeanKernel(hidden_dim, n_agents)
            self.att_2 = MeanKernel(hidden_dim, n_agents)

        self.q_net = QNet(hidden_dim, n_actions)

    def forward(self, x, mask, n_agents=None):

        x = torch.tensor(x).float()
        mask = torch.tensor(mask).float()
        x_enc = self.encoder(x)

        if n_agents == 1:
            x_enc = x_enc.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            x_enc = x_enc.unsqueeze(1)

        h2 = self.att_1(x_enc, mask, n_agents)
        q = self.q_net(h2)

        return q