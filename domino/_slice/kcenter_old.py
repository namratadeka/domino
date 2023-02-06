from __future__ import annotations

import warnings
from functools import wraps
from typing import Union

import numpy as np
import meerkat as mk
from tqdm.auto import tqdm
import torch
from torch.distributions import MultivariateNormal
from sklearn.decomposition import PCA
from scipy.special import logsumexp
from scipy.spatial import distance_matrix
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_gaussian_covariances_full, 
    _estimate_log_gaussian_prob
)


from domino.utils import convert_to_numpy, unpack_args

from .abstract import Slicer


class KCenterSlicer(Slicer):
    def __init__(
        self,
        n_iter: int, 
        n_components: int, 
        n_pca_components: Union[int, None] = 128,
        y_weight: float = 100,
        y_hat_weight: float = 100,
        loss_weight: float = 100,
        append_loss: bool = True,
        init_conf: bool = False
    ):
        super().__init__(n_slices=n_components)
        self.config.n_iter = n_iter
        self.config.n_components = n_components
        self.config.n_pca_components = n_pca_components
        self.config.y_weight = y_weight
        self.config.y_hat_weight = y_hat_weight
        self.config.loss_weight = loss_weight
        self.config.append_loss = append_loss
        self.config.init_conf = init_conf

        if self.config.n_pca_components is None:
            self.pca = None
        else:
            self.pca = PCA(n_components=self.config.n_pca_components)

    def fit(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs",
        losses: Union[str, np.ndarray] = "loss",
        tol: Union[float, None] = 1e-3
    ):
        embeddings, targets, pred_probs, losses = unpack_args(
            data, embeddings, targets, pred_probs, losses
        )
        embeddings, targets, pred_probs, losses = convert_to_numpy(
            embeddings, targets, pred_probs, losses
        )
        if self.config.append_loss:
            embeddings = np.concatenate([embeddings, 
                                        self.config.loss_weight*losses.reshape(-1,1)], 
                                        axis=1)
        else:
            embeddings = np.concatenate([embeddings, 
                                        self.config.y_weight*targets.reshape(-1,1), 
                                        self.config.y_hat_weight*pred_probs.reshape(-1,1)], 
                                        axis=1)
        if self.pca is not None:
            self.pca.fit(X=embeddings)
            embeddings = self.pca.transform(X=embeddings)

        if self.config.init_conf:
            given_centers = self._initialize_centers(targets, pred_probs)
        else:
            given_centers = np.array([])

        prev_logprob_norm = -np.infty
        logprob_norms = []
        for i in tqdm(range(self.config.n_iter)):
            centers, center_idx = self.kcenter_approx(
                                    embeddings=embeddings, 
                                    n_components=self.config.n_components,
                                    given_centers=given_centers
                                )
            given_centers = center_idx
        
            self.centers = centers
            self.center_idx = center_idx
            self._set_params(embeddings)

            logprobs = self.predict_proba(data, 'clip(image)', targets, pred_probs, losses)
            logprob_norm = np.mean(logsumexp(logprobs, axis=1))
            logprob_norms.append(logprob_norm)
            change = logprob_norm - prev_logprob_norm
            if np.abs(change) < tol:
                converged_ = True
                break
        
        return logprob_norms

    def greedy_kcenter(
        self,
        embeddings: np.ndarray,
        n_components: int,
        given_centers: np.ndarray = np.array([])
    ):
        n = embeddings.shape[0]
        d_mat = distance_matrix(embeddings, embeddings)
        if n_components == 0:
            cluster_centers = np.array([], dtype=int)
        else:
            if given_centers.size == 0:
                cluster_centers = np.random.choice(n, 1, replace=False)
                kk = 1
            else:
                cluster_centers = given_centers
                kk = 0

            distance_to_closest = np.amin(d_mat[np.ix_(cluster_centers, np.arange(n))], axis=0)
            while kk < n_components:
                temp = np.argmax(distance_to_closest)
                cluster_centers = np.append(cluster_centers, temp)
                distance_to_closest = np.amin(np.vstack((distance_to_closest, d_mat[temp,:])), axis=0)
                kk += 1

            cluster_centers_idx = cluster_centers[given_centers.size:]

        return embeddings[cluster_centers_idx], cluster_centers_idx
    
    def kcenter_approx(
        self,
        embeddings: np.ndarray,
        n_components: int,
        **_ignored
    ):
        n = embeddings.shape[0]
        d_mat = distance_matrix(embeddings, embeddings)
        cluster_centers = [np.random.randint(n)]
        assignments = np.zeros(n)

        for k in range(n_components-1):
            h = 0
            v = None
            for idx, center in enumerate(cluster_centers):
                cluster_points = np.where(assignments == idx)[0]
                farthest_idx = d_mat[center, cluster_points].argmax()
                farthest_dist = d_mat[center, cluster_points[farthest_idx]]
                if farthest_dist > h:
                    h = farthest_dist
                    v = cluster_points[farthest_idx]
            cluster_centers.append(v)
            assignments = d_mat[np.ix_(cluster_centers, np.arange(n))].argmin(axis=0)
        
        return embeddings[cluster_centers], cluster_centers

    def predict(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs", 
        losses: Union[str, np.ndarray] = "loss"
    ):
        logprobs = self.predict_proba(data, embeddings, targets, pred_probs, losses)
        pred = np.zeros_like(logprobs)
        pred[np.arange(logprobs.shape[0]), np.argmax(logprobs, axis=1)] = 1

        return pred, logprobs

    def predict_proba(
        self,
        data: Union[dict, mk.DataPanel] = None,
        embeddings: Union[str, np.ndarray] = "embedding",
        targets: Union[str, np.ndarray] = "target",
        pred_probs: Union[str, np.ndarray] = "pred_probs", 
        losses: Union[str, np.ndarray] = "loss"
    ):
        embeddings, targets, pred_probs, losses = unpack_args(
            data, embeddings, targets, pred_probs, losses
        )
        embeddings, targets, pred_probs, losses = convert_to_numpy(
            embeddings, targets, pred_probs, losses
        )
        if self.config.append_loss:
            embeddings = np.concatenate([embeddings, 
                                        self.config.loss_weight*losses.reshape(-1,1)], 
                                        axis=1)
        else:
            embeddings = np.concatenate([embeddings, 
                                        self.config.y_weight*targets.reshape(-1,1), 
                                        self.config.y_hat_weight*pred_probs.reshape(-1,1)], 
                                        axis=1)
        if self.pca is not None:
            embeddings = self.pca.transform(X=embeddings)

        logprobs = _estimate_log_gaussian_prob(embeddings, self.centers, self.precisions_chol, 'full')
        
        return logprobs

    def _set_params(
        self,
        embeddings: np.ndarray
    ):
        d_mat = distance_matrix(embeddings, self.centers)
        pred = np.argmin(d_mat, axis=1)
        resp = np.zeros_like(d_mat)
        resp[np.arange(len(d_mat)), pred] = 1
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        reg_covar = 1e-6

        covariances = _estimate_gaussian_covariances_full(resp, embeddings, nk, self.centers, reg_covar)
        self.precisions_chol = _compute_precision_cholesky(covariances, 'full')

    def _initialize_centers(
        self,
        targets: np.ndarray,
        pred_probs: np.ndarray,
        threshold: float = 0.5
    ):
        y_hat = np.where(pred_probs > threshold, 1, 0)
        tp = np.where(y_hat * targets == 1)[0]
        tn = np.where(y_hat + targets + 1 == 1)[0]
        fp = np.where(targets - y_hat == -1)[0]
        fn = np.where(targets - y_hat == 1)[0]
        
        given_centers = np.array([
            np.random.choice(tp),
            np.random.choice(tn),
            np.random.choice(fp),
            np.random.choice(fn)
        ])

        return given_centers
