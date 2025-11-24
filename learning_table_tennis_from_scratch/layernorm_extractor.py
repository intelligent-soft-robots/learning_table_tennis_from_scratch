import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from typing import Dict
import gymnasium as gym


class LayerNormFeaturesExtractor(BaseFeaturesExtractor):
    """
    Creates architecture: Linear(bias=False) → LayerNorm → ReLU for each layer
    """

    def __init__(
        self, 
        observation_space: gym.Space,
        net_arch: list = None,
        use_layer_norm: bool = True,
        hidden_layers_bias: bool = False,
        activation_fn=nn.ReLU,
    ):
        if net_arch is None:
            raise ValueError("net_arch must be specified for LayerNormFeaturesExtractor")

        # Features dim is the output of the last layer
        features_dim = net_arch[-1] if net_arch else observation_space.shape[0]

        super().__init__(observation_space, features_dim)

        if use_layer_norm:
            post_linear_modules = [nn.LayerNorm]
        else:
            post_linear_modules = []

        mlp_layers = create_mlp(
            input_dim=observation_space.shape[0],
            output_dim=features_dim,
            net_arch=net_arch[:-1],
            activation_fn=activation_fn,
            with_bias=hidden_layers_bias,
            post_linear_modules=post_linear_modules
        )
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, observations):
        return self.mlp(observations)
        