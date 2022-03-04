import torch
import torch.nn as nn

class DGN(nn.Module):
    """
    The entire DGN model, capable of using both an Attention and a MeanKernel for propagating information along
    neighboring agents.
    """
    def __init__(self, n_agents, input_dim, hidden_dim, n_actions, use_att=True):
        """
        Initializes the DGN model.
        Args:
            n_agents: max or usual number of agents in the environment, that will use this model
            input_dim: dimension of (flattened) input observations, e.g. something like (13*13*5) = 507
            hidden_dim: dimension of hidden representations
            n_actions: number of actions to generate Q-Values for
            use_att: Whether we will use the Attention Kernel for propagation of information. Otherwise use the Mean
                     Kernel.
        """
        super(DGN, self).__init__()
        self.use_att = use_att
        self.encoder = Encoder(input_dim, hidden_dim)

        if use_att:
            self.att_1 = AttKernel(n_agents, n_heads=4, hidden_dim=hidden_dim)
        else:
            self.att_1 = MeanKernel(hidden_dim, n_agents)
            self.att_2 = MeanKernel(hidden_dim, n_agents)

        self.q_net = QNet(hidden_dim * 3 if use_att else hidden_dim * 2, n_actions)

    def forward(self, x, mask, n_agents=None):
        """
        Performs the forward pass of the DGN. Note that batch_size denotes some number of agents, as we do batching
        by simply having more singular agents and their data present.
        Args:
            x: tensor or numpy array of shape (input_dim,) or (batch_size, input_dim) containing observations for
               one or more agents
            mask: tensor or numpy array of shape (max_neighbors, n_agents) or (batch_size, max_neighbors, n_agents)
                  containing adjacencies for one or more agents
            n_agents: number of agents to make a prediction for
        """
        x = torch.tensor(x).float()
        mask = torch.tensor(mask).float()
        x_enc = self.encoder(x)

        if n_agents == 1:
            x_enc = x_enc.unsqueeze(0)
            mask = mask.unsqueeze(0)
        else:
            x_enc = x_enc.unsqueeze(1)

        h1, h2 = self.att_1(x_enc, mask, n_agents)

        x_enc = x_enc.squeeze(1)

        q = self.q_net(torch.cat((x_enc, h1, h2), dim=1)) if self.use_att else self.q_net(torch.cat((x_enc, h1), dim=1))

        return q

class Encoder(nn.Module):
    """
    Encodes observations.
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Initializes the encoder.

        Args:
            input_dim: input dimension, e.g. something like (13*13*5) = 507
            hidden_dim: the hidden dimension to encode into
        """
        super(Encoder, self).__init__()
        self.lin = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.lin(x))
        return x


class AttKernel(nn.Module):
    """
    The Attention Kernel, that uses nn.MutliheadAttention to distribute information
    """
    def __init__(self, n_agents, n_heads, hidden_dim):
        """
        Initialize the Attention Kernel.
        Args:
            n_agents: max or usual number of agents in the environment, that will use this model
            n_heads: number of attention heads
            hidden_dim: hidden dimension of the encoded observations
        """
        super(AttKernel, self).__init__()
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.att_1 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.att_2 = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(self, x, mask, n_agents=None):
        """
        Performs the forward pass by broadcasting the features to the size of the adjacencies and performing
        matrix multiplication as desired by the paper. Then performs MultiheadAttention.
        Args:
            x: tensor of shape (n_agents, hidden_dim) or (n_agents, 1, hidden_dim) where n_agents can be any number of
               agents to allow for both singular predictions and batching. Contains encoded observations for each agent
            mask: tensor of shape (n_agents, max_neighbors, hidden_dim) representing the adjacencies for each agent
            n_agents: number of agents to make a prediction for
        Returns:
            x: tensor of shape (n_agents, hidden_dim) containing updated hidden representations for each agent
        """
        n_agents = n_agents if n_agents else self.n_agents

        feature_broadcast = torch.broadcast_to(x, (n_agents, self.n_agents, self.hidden_dim))
        x_feats = torch.bmm(mask, feature_broadcast)

        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        x_feats_1, _ = self.att_1(x, x_feats, x_feats)
        x_feats_2, _ = self.att_2(x, x_feats_1, x_feats_1)

        return x_feats_1.squeeze(1), x_feats_2.squeeze(1)


class MeanKernel(nn.Module):
    """
    The MeanKernel that uses the mean operation to distribute information.
    """
    def __init__(self, hidden_dim, n_agents):
        """
        Initializes the MeanKernel.
        Args:
            hidden_dim: dimension of hidden representations
            n_agents: max or usual number of agents in the environment, that will use this model
        """
        super(MeanKernel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_agents = n_agents

    def forward(self, x, mask, n_agents=None):
        """
        Perform the forward pass with the mean operation by again broadcasting the features and performing mean along
        adjacent representations.
        Args:
            x: tensor of shape (n_agents, hidden_dim) or (n_agents, 1, hidden_dim) where n_agents can be any number of
               agents to allow for both singular predictions and batching. Contains encoded observations for each agent
            mask: tensor of shape (n_agents, max_neighbors, hidden_dim) representing the adjacencies for each agent
            n_agents: number of agents to make a prediction for
        Returns:
            x: tensor of shape (n_agents, hidden_dim) containing updated hidden representations for each agent
        """
        n_agents = n_agents if n_agents else self.n_agents

        feature_broadcast = torch.broadcast_to(x, (n_agents, self.n_agents, self.hidden_dim))
        x_feats = torch.bmm(mask, feature_broadcast)
        x_mean = torch.mean(x_feats, dim=1)

        return x_mean, None


class QNet(nn.Module):
    """
    Final model part calculating Q-Values.
    """
    def __init__(self, hidden_dim, n_actions):
        """
        Initialize the Q-Net
        Args:
            hidden_dim: dimension of hidden representations
            n_actions: number of actions to generate Q-Values for
        """
        super(QNet, self).__init__()
        self.fc = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        """
        Performs the forward pass as a simple linear layer.
        Args:
            x: tensor of shape (n_agents, hidden_dim) with hidden representations of all agents
        Returns:
            q: tensor of shape (n_agents, n_actions) containing Q-Values for all actions for each agent
        """
        q = self.fc(x)
        return q
