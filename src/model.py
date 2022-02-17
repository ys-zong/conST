#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .layers import AvgReadout, Discriminator, Clusterator, Discriminator_cluster, full_block, GraphConvolution, \
    InnerProductDecoder


class conST(nn.Module):
    def __init__(self, input_dim, params, n_clusters, dim, use_img):
        super(conST, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2 + params.feat_hidden2
        self.tau = 0.5
        self.n_clusters = n_clusters
        self.dim = dim
        self.params = params
        self.use_img = use_img

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # img feature autoencoder
        if self.use_img:
            self.img_encoder = nn.Sequential()
            self.img_encoder.add_module('img_encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
            self.img_encoder.add_module('img_encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

            self.img_decoder = nn.Sequential()
            self.img_decoder.add_module('img_decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))

        # GCN layers
        if self.use_img:
            self.gc1 = GraphConvolution(params.feat_hidden2 * 2, params.gcn_hidden1, params.p_drop, act=F.relu)
        else:
            self.gc1 = GraphConvolution(params.feat_hidden2, params.gcn_hidden1, params.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)

        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # projection
        self.fc1 = torch.nn.Linear(params.feat_hidden2, params.feat_hidden2 * 2)
        self.fc2 = torch.nn.Linear(params.feat_hidden2 * 2, params.feat_hidden2)

        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.cluster = Clusterator(params.feat_hidden2, K=self.n_clusters)
        self.disc_c = Discriminator_cluster(params.feat_hidden2, params.feat_hidden2, n_nb=self.dim,
                                            num_clusters=self.n_clusters)
        self.disc = Discriminator(params.feat_hidden2)

    def encode(self, x, adj, img=None):
        feat_x = self.encoder(x)
        if self.use_img:
            feat_img = self.img_encoder(img)
            feat = torch.cat((feat_x, feat_img), 1)
            hidden1 = self.gc1(feat, adj)
            return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat, feat_x, feat_img
        else:
            hidden1 = self.gc1(feat_x, adj)
            return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj, img=None):
        if self.use_img is False:
            mu, logvar, feat_x = self.encode(x, adj)
            gnn_z = self.reparameterize(mu, logvar)
            z = torch.cat((feat_x, gnn_z), 1)
            de_feat = self.decoder(z)

            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            return z, mu, logvar, de_feat, q, feat_x, gnn_z
        else:
            mu, logvar, feat, feat_x, feat_img = self.encode(x, adj, img)
            gnn_z = self.reparameterize(mu, logvar)
            z = torch.cat((feat_x, gnn_z, feat_img), 1)
            de_feat = self.decoder(z)
            de_feat_img = self.img_decoder(z)

            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

            return z, mu, logvar, de_feat, de_feat_img, q, feat, gnn_z

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def cont_l2l(self, z1: torch.Tensor, z2: torch.Tensor,
                 mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def l2c_forward(self, h_1, h_2, cluster_temp):
        Z, S = self.cluster(h_1, cluster_temp)
        Z_t = S @ Z
        c2 = Z_t
        c2 = self.sigm(c2)

        ret = self.disc_c(c2, c2, h_1, h_1, h_2, S, None, None)
        return ret

    def l2g_forward(self, h_1, h_2):
        c = self.read(h_1, msk=None)
        c = self.sigm(c)
        c_x = c.unsqueeze(1)
        c_x = c_x.expand_as(h_1)
        ret = self.disc(c_x, h_1, h_2, None, None)
        return ret

    def cont_bxent(self, lbl, logits):
        b_xent = nn.BCEWithLogitsLoss()
        logits = logits.reshape(1, -1)
        cont_bxent = b_xent(logits, lbl)
        return cont_bxent
