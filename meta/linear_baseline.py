import torch
import numpy as np
import torch.nn as nn
from misc.rl_utils import get_return


class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in Duan et al., 2016
    Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel,
    "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016
    Args:
        input_size (int): Input size from environment
        args (argparse): Python argparse that contains arguments
        reg_coeff (float): Regularization coefficient. Default: 1e-5
    Reference:
        https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/baseline.py
    """
    def __init__(self, input_size, args, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()

        self.input_size = input_size
        self.args = args
        self._reg_coeff = reg_coeff

        self.weight = nn.Parameter(torch.Tensor(self.feature_size,), requires_grad=False)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size, dtype=torch.float32, device=self.weight.device)

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, obs):
        batch_size, sequence_length, _ = obs.shape
        assert sequence_length == self.args.ep_horizon, "Length should equal to episodic horizon"

        ones = torch.ones((sequence_length, batch_size, 1))
        obs = obs.clone().transpose(0, 1)
        time_step = torch.arange(sequence_length).view(-1, 1, 1) * ones / 100.0

        return torch.cat([
            obs,
            obs ** 2,
            time_step,
            time_step ** 2,
            time_step ** 3,
            ones
        ], dim=2)

    def fit(self, obs, return_):
        # featmat.shape: sequence_length * batch_size x feature_size
        featmat = self._feature(obs).view(-1, self.feature_size)

        # returns.shape: sequence_length * batch_size x 1
        returns = return_.view(-1, 1)

        reg_coeff = self._reg_coeff
        XT_y = torch.matmul(featmat.t(), returns)
        XT_X = torch.matmul(featmat.t(), featmat)
        for _ in range(5):
            try:
                coeffs, _ = torch.lstsq(XT_y, XT_X + reg_coeff * self._eye)
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            raise RuntimeError(
                'Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.weight.copy_(coeffs.flatten())

    def forward(self, obs, reward):
        # Fit linear feature baseline
        obs = torch.from_numpy(np.stack(obs, axis=1)).float()
        return_ = get_return(reward, self.args)
        self.fit(obs, return_)

        # Return value
        features = self._feature(obs)
        value = torch.mv(features.view(-1, self.feature_size), self.weight)

        return value.view(features.shape[:2]).transpose(0, 1)
