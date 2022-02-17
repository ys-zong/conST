#
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
from progress.bar import Bar
from sklearn.cluster import KMeans
from src.model import conST
from torch_geometric.utils import dropout_adj
from torch_sparse import SparseTensor


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn


def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask

    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)

    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


class conST_training:
    def __init__(self, node_X, graph_dict, params, n_clusters, img = None):
        self.params = params
        self.device = params.device
        self.epochs = params.epochs
        self.node_X = torch.FloatTensor(node_X.copy()).to(self.device)

        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)
        self.norm_value = graph_dict["norm_value"]
        
        self.n_clusters = n_clusters
        self.dim = self.adj_norm.shape[0]

        if img is not None:
            self.use_img = True
            self.img = torch.FloatTensor(img).to(self.device)
        else:
            self.use_img = False
            self.img = None

        if params.using_mask is True:
            self.adj_mask = graph_dict["adj_mask"].to(self.device)
        else:
            self.adj_mask = None

        self.model = conST(self.params.cell_feat_dim, self.params, self.n_clusters, self.dim, self.use_img).to(self.device)
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()),
                                          lr=self.params.gcn_lr, weight_decay=self.params.gcn_decay)

        if self.use_img is False:
            self.fc1 = torch.nn.Linear(self.params.feat_hidden2, self.params.feat_hidden2*2)
            self.fc2 = torch.nn.Linear(self.params.feat_hidden2*2, self.params.feat_hidden2)
        else:
            self.fc1 = torch.nn.Linear(self.params.feat_hidden2 * 2 + self.params.gcn_hidden2, self.params.feat_hidden2 * 3)
            self.fc2 = torch.nn.Linear(self.params.feat_hidden2 * 3, self.params.feat_hidden2 * 2 + self.params.gcn_hidden2)

        self.beta = params.beta

    def pretraining(self):
        self.model.train()
        bar = Bar('Pretraining stage: ', max=self.epochs)
        bar.check_tty = False
        for epoch in range(self.epochs):
            start_time = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            if self.use_img is False:
                latent_z, mu, logvar, de_feat, _, feat_x, _ = self.model(self.node_X, self.adj_norm)

                loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,
                                    logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value, mask=self.adj_label)
                loss_rec = reconstruction_loss(de_feat, self.node_X)
                loss = self.params.feat_w * loss_rec + self.params.gcn_w * loss_gcn
            else:
                latent_z, mu, logvar, de_feat, de_feat_img, _, feat_x, _ = self.model(self.node_X, self.adj_norm,
                                                                                      self.img)

                loss_gcn = gcn_loss(preds=self.model.dc(latent_z), labels=self.adj_label, mu=mu,
                                    logvar=logvar, n_nodes=self.params.cell_num, norm=self.norm_value,
                                    mask=self.adj_label)
                loss_rec = reconstruction_loss(de_feat, self.node_X)
                loss_rec_img = reconstruction_loss(de_feat_img, self.img)
                loss = self.params.feat_w * loss_rec + self.params.gcn_w * loss_gcn + self.params.img_w * loss_rec_img
            loss.backward()
            self.optimizer.step()

            end_time = time.time()
            batch_time = end_time - start_time
            bar_str = '{} / {} | Left time: {batch_time:.2f} mins| Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch + 1, self.epochs,
                                        batch_time=batch_time * (self.epochs - epoch) / 60, loss=loss.item())
            bar.next()
        bar.finish()

    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'], strict = False)

    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z = self.model(self.node_X, self.adj_norm)
        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()
        return latent_z, q, feat_x, gnn_z

    def get_embedding(self):
        self.model.eval()
        embedding, _, _, _, q, feat_x, gnn_z = self.model(self.node_X, self.adj_norm)
        embedding = embedding.data.cpu().numpy()
        return embedding
    
    def drop_edge(self, adj, drop_rate):
        adj = adj.detach().cpu().to_dense()
        edge_index = adj.nonzero(as_tuple=False).t()   
        edge_index = dropout_adj(edge_index, p=drop_rate)[0]
        edge_weight = adj[edge_index[0], edge_index[1]]
        droped_adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=adj.shape)
        droped_adj = droped_adj.to_torch_sparse_coo_tensor().to(self.device)
        return droped_adj
    
    def augment_edge_node(self, edge_drop_p1, edge_drop_p2, node_drop_p1, node_drop_p2):
        adj1 = self.drop_edge(self.adj_norm, edge_drop_p1)
        adj2 = self.drop_edge(self.adj_norm, edge_drop_p2)
        
        x_1 = drop_feature(self.node_X, node_drop_p1)
        x_2 = drop_feature(self.node_X, node_drop_p2)
        if self.use_img is True:
            img_1 = drop_feature(self.img, node_drop_p1)
            img_2 = drop_feature(self.img, node_drop_p1)
            return adj1, adj2, x_1, x_2, img_1, img_2
        return adj1, adj2, x_1, x_2

    def augment_l2cg(self, node, adj):
        # shuffle the nodes
        idx = np.random.permutation(node.shape[0])
        shuf_fts = self.node_X[idx, :]

        # change adj
        adj_dropped = self.drop_edge(adj, 0.8)
        return shuf_fts, adj_dropped

    def major_training(self):
        # initialize cluster parameter
        kmeans = KMeans(n_clusters=self.params.dec_cluster_n, n_init=self.params.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()

        bar = Bar('Major training stage: ', max=self.epochs)
        bar.check_tty = False
        for epoch_id in range(self.epochs):
            # DEC clustering update
            if epoch_id % self.params.dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < self.params.dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', self.params.dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # training model
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            if self.use_img is False:
                adj1, adj2, x_1, x_2 = self.augment_edge_node(self.params.edge_drop_p1,
                                                                            self.params.edge_drop_p2,
                                                                            self.params.node_drop_p1,
                                                                            self.params.node_drop_p2)
                # local-to-local

                z1, mu1, logvar1, de_feat1, q1, feat_x1, gnn_z1 = self.model(x_1, adj1)
                z2, mu2, logvar2, de_feat2, q2, feat_x2, gnn_z2 = self.model(x_2, adj2)
                feat_x1 = self.model.projection(feat_x1)
                feat_x2 = self.model.projection(feat_x2)
                loss_cont = self.model.cont_l2l(feat_x1, feat_x2)

                latent_z, mu, logvar, de_feat, out_q, feat_x,  _ = self.model(self.node_X, self.adj_norm)

                shuf_fts, adj_changed = self.augment_l2cg(self.node_X, self.adj_norm)

                latent_z3, mu3, logvar3, de_feat3, out_q3, feat_x3, gnn_z3 = self.model(shuf_fts, adj_changed)

                ret_l2c = self.model.l2c_forward(feat_x, feat_x3, self.beta)
                ret_l2g = self.model.l2g_forward(feat_x, feat_x3)

                lbl_1 = torch.ones(1, ret_l2c.shape[1] // 2)
                lbl_2 = torch.zeros(1, ret_l2c.shape[1] // 2)
                lbl = torch.cat((lbl_1, lbl_2), 1).to(self.params.device)

                cont_l2c = self.model.cont_bxent(lbl, ret_l2c)
                cont_l2g = self.model.cont_bxent(lbl, ret_l2g)

                loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss = self.params.dec_kl_w * loss_kl + self.params.cont_l2l * loss_cont + \
                       self.params.cont_l2g * cont_l2g + self.params.cont_l2c + cont_l2c
            else:
                adj1, adj2, x_1, x_2, img_1, img_2 = self.augment_edge_node(self.params.edge_drop_p1,
                                                              self.params.edge_drop_p2,
                                                              self.params.node_drop_p1,
                                                              self.params.node_drop_p2)
                z1, mu1, logvar1, de_feat1, de_feat_img1, q1, feat_x1, gnn_z1 = self.model(x_1, adj1, img_1)
                z2, mu2, logvar2, de_feat2, de_feat_img2, q2, feat_x2, gnn_z2 = self.model(x_2, adj2, img_2)
                feat_x1 = self.model.projection(z1)
                feat_x2 = self.model.projection(z2)
                loss_cont = self.model.contrastive_loss(feat_x1, feat_x2)

                latent_z, mu, logvar, de_feat, de_feat_img, out_q, feat_x, _ = self.model(self.node_X,
                                                                                          self.adj_norm, self.img)

                shuf_fts, adj_changed = self.augment_l2cg(self.node_X, self.adj_norm)

                latent_z3, mu3, logvar3, de_feat3, de_feat_img, out_q3, feat_x3, gnn_z3 = self.model(shuf_fts,
                                                                                                     self.adj_norm, self.img)
                ret_l2c = self.model.l2c_forward(latent_z, latent_z3, self.beta)
                ret_l2g = self.model.l2g_forward(latent_z, latent_z3)

                lbl_1 = torch.ones(1, ret_l2c.shape[1] // 2)
                lbl_2 = torch.zeros(1, ret_l2c.shape[1] // 2)
                lbl = torch.cat((lbl_1, lbl_2), 1).to(self.params.device)

                cont_l2c = self.model.cont_bxent(lbl, ret_l2c)
                cont_l2g = self.model.cont_bxent(lbl, ret_l2g)

                loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
                loss = self.params.dec_kl_w * loss_kl + self.params.cont_l2l * loss_cont + \
                       self.params.cont_l2g * cont_l2g + self.params.cont_l2c + cont_l2c

            loss.backward()
            self.optimizer.step()

            bar_str = '{} / {} | Loss: {loss:.4f}'
            bar.suffix = bar_str.format(epoch_id + 1, self.epochs, loss=loss.item())
            bar.next()
        bar.finish()