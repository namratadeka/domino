from __future__ import annotations
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial import distance_matrix


class ContrastiveLoss(nn.Module):
    '''
    Code taken from https://github.com/marrrcin/pytorch-simclr-efficientnet.
    '''
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

class ContrastiveEmbeddings(nn.Module):
    def __init__(self, input_dim, enc_dim, batch_size, cont_weight, recon_weight, orth_weight) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 256
        self.enc_dim = enc_dim
        self.batch_size = batch_size

        self.build_modules()
        self.build_opt()
        self.metric_loss = nn.TripletMarginLoss(margin=1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.c_loss_weight = cont_weight
        self.r_loss_weight = recon_weight
        self.cov_penality_weight = orth_weight

    def build_modules(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.enc_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.enc_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )

        for m in self.encoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)
        for m in self.decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight)

    def build_opt(self):
        self.opt = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.0001)

    def forward(self, x_a, x_p, x_n):
        emb_a = self.encoder(x_a)
        emb_p = self.encoder(x_p)
        emb_n = self.encoder(x_n)
        hat_x_a = self.decoder(emb_a)
        hat_x_p = self.decoder(emb_p)
        hat_x_n = self.decoder(emb_n)

        c_loss = self.metric_loss(emb_a, emb_p, emb_n)
        r_loss = F.mse_loss(hat_x_a, x_a) + F.mse_loss(hat_x_p, x_p) + F.mse_loss(hat_x_n, x_n)

        cov = torch.cov(torch.vstack([emb_a, emb_p, emb_n]).T)
        cov_penalty = cov.sum() - torch.diag(cov).sum()

        loss = self.r_loss_weight*r_loss \
             + self.c_loss_weight*c_loss \
             + self.cov_penality_weight*torch.abs(cov_penalty)

        reg = 1e-6
        orth_loss = torch.tensor(0)
        for name, param in self.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0]).to(self.device)
                orth_loss = orth_loss + (reg * sym.abs().sum())
        loss += orth_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return c_loss.item(), r_loss.item(), cov_penalty.item()

    def fit(self, x, y, y_hat):
        x_a, x_p, x_n = self.mine(x, y, y_hat)
        for epoch in range(500):
            idx = np.random.choice(len(x_a), len(x_a), replace=False)
            iterator = tqdm(range(0, x_a.shape[0], self.batch_size))
            for i in iterator:
                c_loss, r_loss, cov = self.forward(x_a[idx][i : i+self.batch_size], 
                                                   x_p[idx][i : i+self.batch_size], 
                                                   x_n[idx][i : i+self.batch_size]
                                        )
                iterator.set_description('epoch {} | c_loss: {:.4f} | r_loss: {:.4f} | cov: {:.4f}'.format(epoch, c_loss, r_loss, cov))

    @torch.no_grad()
    def transform(self, x):
        x = torch.FloatTensor(x).to(self.device)
        emb = self.encoder(x)
        return emb.cpu().numpy()

    def mine(self, x, y, y_hat):
        pred = np.where(y_hat >= 0.5, 1, 0)
        error_class = np.zeros((len(y)))
        t0 = np.where(y == 0)[0]
        t1 = np.where(y == 1)[0]
        p0 = np.where(pred == 0)[0]
        p1 = np.where(pred == 1)[0]
        error_class[np.array(list(set(t0).intersection(set(p0))))] = 0
        error_class[np.array(list(set(t0).intersection(set(p1))))] = 1
        error_class[np.array(list(set(t1).intersection(set(p0))))] = 2
        error_class[np.array(list(set(t1).intersection(set(p1))))] = 3

        fp = x[error_class == 1]
        fn = x[error_class == 2]
        tp = x[error_class == 3]
        tn = x[error_class == 0]

        # y = 1
        fn_dmat = distance_matrix(fn, fn)
        fn_tp_dmat = distance_matrix(fn, tp)
        fn_anchors = fn
        fn_positives = fn[fn_dmat.argmax(axis=1)]
        fn_negatives = tp[fn_tp_dmat.argmin(axis=1)]

        # y = 0
        fp_dmat = distance_matrix(fp, fp)
        fp_tn_dmat = distance_matrix(fp, tn)
        fp_anchors = fp
        fp_positives = fp[fp_dmat.argmax(axis=1)]
        fp_negatives = tn[fp_tn_dmat.argmin(axis=1)]

        anchors = np.concatenate([fn_anchors, fp_anchors], axis=0)
        positives = np.concatenate([fn_positives, fp_positives], axis=0)
        negatives = np.concatenate([fn_negatives, fp_negatives], axis=0)

        return (torch.FloatTensor(anchors).to(self.device),
                torch.FloatTensor(positives).to(self.device),
                torch.FloatTensor(negatives).to(self.device))

def numpy_combinations(x):
    idx = np.stack(np.triu_indices(len(x), k=1), axis=-1)
    return x[idx]
