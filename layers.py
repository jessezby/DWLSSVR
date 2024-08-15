"""
SinkhornDistance Module

This module implements the Sinkhorn Distance algorithm for computing an approximation
of the regularized Optimal Transport (OT) cost between two empirical measures.
"""

import torch
import torch.nn as nn

class SinkhornDistance(nn.Module):
    """
    Computes an approximation of the regularized OT cost for point clouds.

    Given two empirical measures with P_1 locations x ∈ ℝ^(D_1) and P_2 locations y ∈ ℝ^(D_2),
    this module outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): Regularization coefficient.
        max_iter (int): Maximum number of Sinkhorn iterations.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'none'.

    Shape:
        - Input: (N, P_1, D_1), (N, P_2, D_2)
        - Output: (N) or (), depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        """
        Compute the Sinkhorn distance between two point clouds.

        Args:
            x (torch.Tensor): First point cloud.
            y (torch.Tensor): Second point cloud.

        Returns:
            tuple: (cost, pi, C)
                - cost (torch.Tensor): Sinkhorn distance
                - pi (torch.Tensor): Transport plan
                - C (torch.Tensor): Cost matrix
        """
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        batch_size = x.shape[0] if x.dim() > 2 else 1

        mu = torch.full((batch_size, x_points), 1.0 / x_points, dtype=torch.float).squeeze()
        nu = torch.full((batch_size, y_points), 1.0 / y_points, dtype=torch.float).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        # Sinkhorn iterations
        for _ in range(self.max_iter):
            u1 = u.clone()
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            if err.item() < 1e-1:
                break

        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, u, v))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """
        Compute the modified cost for logarithmic updates.

        M_ij = (-c_ij + u_i + v_j) / ε

        Args:
            C (torch.Tensor): Cost matrix.
            u (torch.Tensor): Dual variable.
            v (torch.Tensor): Dual variable.

        Returns:
            torch.Tensor: Modified cost matrix.
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        """
        Compute the cost matrix between two point clouds.

        Args:
            x (torch.Tensor): First point cloud.
            y (torch.Tensor): Second point cloud.
            p (int, optional): Power for the Euclidean distance. Default: 2.

        Returns:
            torch.Tensor: Cost matrix C_ij = |x_i - y_j|^p.
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        """
        Compute the barycenter between two vectors.

        This subroutine is used for kinetic acceleration through extrapolation.

        Args:
            u (torch.Tensor): First vector.
            u1 (torch.Tensor): Second vector.
            tau (float): Interpolation parameter.

        Returns:
            torch.Tensor: Barycenter of u and u1.
        """
        return tau * u + (1 - tau) * u1