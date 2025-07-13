#!/usr/bin/env python
# coding: utf-8

"""
Goal-Conditioned Supervised Learning (GCSL) for table tennis.
"""

# === Imports ===
import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict, defaultdict, deque
import itertools
import tikzplotlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Optional, Union, Tuple, Any, Type
from copy import deepcopy
import dataclasses
from types import MappingProxyType
from ttr_diffusion_v4 import inference_diffusion_policy, TTRDiffusionDataset, ConditionalUnet1D

# === Load Main Configuration ===
# Assume config.json is in the same directory or accessible path
main_config_path = 'config.json'
if not os.path.exists(main_config_path):
    raise FileNotFoundError(f"Main config file not found at: {main_config_path}")
with open(main_config_path, 'r') as f:
    main_config = json.load(f)

gcsl_config_path = main_config.get("gcsl_config")
if not gcsl_config_path or not os.path.exists(gcsl_config_path):
     raise FileNotFoundError(f"GCSL config path '{gcsl_config_path}' not found or not specified in {main_config_path}")

# === Load GCSL Configuration ===
with open(gcsl_config_path, 'r') as f:
    gcsl_config = json.load(f)

# === Configurations ===
reward_config_file = main_config.get("reward_config")
hysr_one_ball_config_file = main_config.get("hysr_config")
dataset_folder_groups = gcsl_config['paths']['dataset_folder_groups']
data_path = gcsl_config['paths']['data_path']
data_path_2 = gcsl_config['paths']['data_path_eval']
log_file_path = gcsl_config['paths']['log_file_path']
vsgcsl_path = gcsl_config['paths']['vsgcsl_path']
output_folder = gcsl_config['paths']['output_folder'] # Base output folder

# Add vsgcsl_path to sys.path if it's not already there
if vsgcsl_path not in sys.path:
    sys.path.append(vsgcsl_path)

# Import custom modules
from learning_table_tennis_from_scratch.hysr_goal_env import HysrGoalEnv

# Import GCSL components
from gcsl.algo.networks import CBCNetwork, StateGoalNetwork
from gcsl.algo.buffer import ReplayBuffer
from gcsl.policy import GoalConditionedPolicy


class Mode(Enum):
    TRAINING = auto()
    DEBUG = auto()
    VIDEO = auto()

def is_sin_traj(filename):
    if not "d" in filename and not "m" in filename and not "p" in filename:
        return True
    else:
        return False
    
def is_ppo_traj(filename):
    if "ppo" in filename or "test" in filename:
        return True
    else:
        return False
    
def is_peril_traj(filename):
    if "per" in filename:
        return True
    else:
        return False
    
def is_sin_plus_noise_traj(filename):
    if "m" in filename and not "per" in filename and not "ppo" in filename:
        return True
    else:
        return False
    
def show_statistics_of_traj_types_folder(foldername):
    all_files = [f for f in os.listdir(foldername) if f.endswith(".json")]
    show_statistics_of_traj_types(all_files, foldername)


def show_statistics_of_traj_types(file_list, foldername):
    print(f"showing statistics for {len(file_list)} files in {foldername}")
    n_sin = 0
    n_sin_plus_noise = 0
    n_ppo = 0
    n_peril = 0
    n_none = 0
    n_multiple = 0
    for file in file_list:
        filename_complete = os.path.join(foldername, file)
        filename = file 
        if not os.path.isfile(filename_complete):
            continue
        if is_sin_traj(filename):
            n_sin += 1
        if is_sin_plus_noise_traj(filename):
            n_sin_plus_noise += 1
        if is_ppo_traj(filename):
            n_ppo += 1
        if is_peril_traj(filename):
            n_peril += 1
        
        if sum([is_sin_traj(filename), is_sin_plus_noise_traj(filename), is_ppo_traj(filename), is_peril_traj(filename)]) > 1:
            print("multiple", filename)
            n_multiple += 1
        elif sum ([is_sin_traj(filename), is_sin_plus_noise_traj(filename), is_ppo_traj(filename), is_peril_traj(filename)]) == 0:
            print("none", filename)
            n_none += 1

    # print percentages
    n_total = n_sin + n_sin_plus_noise + n_ppo + n_peril + n_none
    print(f"Total: {n_total}")
    print(f"Sin: {n_sin} ({n_sin/n_total*100:.2f}%)")
    print(f"Sin + Noise: {n_sin_plus_noise} ({n_sin_plus_noise/n_total*100:.2f}%)")
    print(f"PPO: {n_ppo} ({n_ppo/n_total*100:.2f}%)")
    print(f"Peril: {n_peril} ({n_peril/n_total*100:.2f}%)")
    print(f"None: {n_none} ({n_none/n_total*100:.2f}%)")
    print(f"Multiple: {n_multiple} ({n_multiple/n_total*100:.2f}%)")

# Table tennis constants
table_center = [0.4, 1.57, 0.755]
tc = table_center
half_table_size = [0.7625, 1.37]
hts = half_table_size
center_goal = [tc[0] + hts[0], tc[1], tc[2]]

# === Environment Setup ===
def create_environment():
    """
    Creates and returns the table tennis environment.
    """
    env = HysrGoalEnv(
        reward_config_file=reward_config_file,
        hysr_one_ball_config_file=hysr_one_ball_config_file
    )
    return env


class DataNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Prevent division by zero
        self.std[self.std == 0] = 1.0  
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean



class NNAgentDeterministic(nn.Module):
    """
    Neural Network Agent for GCSL.
    """
    def __init__(self, env, n_hidden=1024, n_layers=2):
        super().__init__()
        layers = [nn.Linear(26, n_hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.ReLU()]
        layers.append(nn.Linear(n_hidden, 8))  # Output layer
        self.net = nn.Sequential(*layers)
      
    def forward(self, state, goal, horizon):
        x = torch.cat([state, goal, horizon], -1)
        return self.net(x)
      
    def get_action(self, state, goal, horizon, greedy=False):
        state_torch = torch.tensor(state['observation'], dtype=torch.float32).unsqueeze(0)
        goal_torch = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
        horizon_torch = torch.tensor([horizon], dtype=torch.float32).unsqueeze(0)
        action = self.forward(state_torch, goal_torch, horizon_torch).detach().numpy()[0]
        return action



class NNAgent(nn.Module, GoalConditionedPolicy):
    """
    Neural Network Agent for GCSL with probabilistic policy.
    """
    def __init__(self, env, n_hidden=1024, n_layers=2):
        super().__init__()
        self.action_dim = env.action_space.shape[0]

        self.net = StateGoalNetwork(
            env=env,
            dim_out=self.action_dim * 2,  # Output mean and log_std
            layers=[n_hidden] * n_layers,
            add_extra_conditioning=True
        )
      
    def forward(self, state, goal, horizon=None):
        mean_log_std = self.net(state, goal, horizon=horizon)
        mean, log_std = torch.chunk(mean_log_std, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
      
    def get_action(self, state, goal, horizon=0, greedy=False):
        state_torch = torch.tensor(state['observation'], dtype=torch.float32).unsqueeze(0)
        goal_torch = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
        horizon_torch = torch.tensor([[horizon]], dtype=torch.float32)
        mean, log_std = self.forward(state_torch, goal_torch, horizon_torch)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        if greedy:
            action = mean.detach().numpy()[0]
        else:
            action = dist.sample().detach().numpy()[0]
        return action

    def nll(self, obs, goal, actions, horizon=None):
        mean, log_std = self.forward(obs, goal, horizon)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(actions).sum(-1)  # Sum over action dimensions
        return nll

    def entropy(self, obs, goal, horizon=None):
        mean, log_std = self.forward(obs, goal, horizon)
        return (0.5 + 0.5 * np.log(2 * np.pi)) * mean.size(-1) + log_std.sum(-1)
    

class CustomCBCNetwork(nn.Module):
    def __init__(self, dim_input, dim_conditioning, dim_output, layers, 
                 nonlinearity=nn.ReLU, dropout=0.0, use_layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = dim_input + dim_conditioning
        for layer_dim in layers:
            layer_modules = []
            layer_modules.append(('linear', nn.Linear(prev_dim, layer_dim)))
            if use_layer_norm:
                layer_modules.append(('layer_norm', nn.LayerNorm(layer_dim)))
            layer_modules.append(('activation', nonlinearity()))
            if dropout > 0:
                layer_modules.append(('dropout', nn.Dropout(dropout)))
            self.layers.append(nn.ModuleDict(layer_modules))
            prev_dim = layer_dim + dim_conditioning  # Adjust for conditioning
        self.output_layer = nn.Linear(prev_dim, dim_output)
    
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        for layer in self.layers:
            x = layer['linear'](x)
            if 'layer_norm' in layer:
                x = layer['layer_norm'](x)
            x = layer['activation'](x)
            if 'dropout' in layer:
                x = layer['dropout'](x)
            x = torch.cat([x, goal], dim=-1)  # Conditioning
        x = self.output_layer(x)
        return x


class CBCAgent(nn.Module, GoalConditionedPolicy):
    def __init__(self, env, n_hidden=1024, n_layers=2, dropout=0.0, 
                 use_layer_norm=False, data_normalizer=None):
        
        print(f"net params: n_hidden={n_hidden}, n_layers={n_layers}, dropout={dropout}, use_layer_norm={use_layer_norm}")
        super().__init__()
        self.action_dim = env.action_space.shape[0]
        self.data_normalizer = data_normalizer
        
        # Get state and goal dimensions from the environment
        ob_reset, _ = env.reset()
        state_dim = ob_reset['observation'].shape[0]
        goal_dim = ob_reset['desired_goal'].shape[0]
        
        # The output dimension is twice the action dimension for mean and log_std
        output_dim = self.action_dim * 2
        
        if use_layer_norm:
            print("--- Using Layer Norm ---")
            self.net = CustomCBCNetwork(
                dim_input=state_dim,
                dim_conditioning=goal_dim,
                dim_output=output_dim,
                layers=[n_hidden] * n_layers,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
        else:
            print("--- Not Using Layer Norm ---")
            # Create the CBC network
            self.net = CBCNetwork(
                dim_input=state_dim,
                dim_conditioning=goal_dim,
                dim_output=output_dim,
                layers=[n_hidden] * n_layers,
                dropout=dropout,
                add_conditioning=True,  # Enable conditioning at each layer
                nonlinearity=torch.nn.ReLU
            )
      
    def forward(self, state, goal, horizon: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            goal: Goal tensor of shape (batch_size, goal_dim)
            horizon: Optional horizon tensor (not used in this implementation)
            
        Returns:
            Tuple of (mean, log_std) tensors for the action distribution
        """
        output = self.net(state, goal)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical instability
        return mean, log_std
      
    def get_action(self, state, goal, horizon=0, greedy=False):

        state_observation = state['observation']
        if self.data_normalizer is not None:
            state_observation = self.data_normalizer.transform(state_observation)
        state_torch = torch.tensor(state_observation, dtype=torch.float32).unsqueeze(0)
        goal_torch = torch.tensor(goal, dtype=torch.float32).unsqueeze(0)
        
        mean, log_std = self.forward(state_torch, goal_torch)
        
        if greedy:
            action = mean.detach().numpy()[0]
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample().detach().numpy()[0]
        
        return action

    def nll(self, obs, goal, actions, horizon=None):
        """
        Compute the negative log likelihood of actions given states and goals.
        
        Args:
            obs: Observation tensor
            goal: Goal tensor
            actions: Action tensor
            horizon: Optional horizon tensor (not used)
            
        Returns:
            Negative log likelihood tensor
        """
        mean, log_std = self.forward(obs, goal)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return -dist.log_prob(actions).sum(-1)  # Sum over action dimensions

    def entropy(self, obs, goal, horizon=None):
        """
        Compute the entropy of the policy distribution.
        
        Args:
            obs: Observation tensor
            goal: Goal tensor
            horizon: Optional horizon tensor (not used)
            
        Returns:
            Entropy tensor
        """
        _, log_std = self.forward(obs, goal)
        return (0.5 + 0.5 * np.log(2 * np.pi)) * self.action_dim + log_std.sum(-1)

# === Data Loading ===
# Modified to accept action_repeat_counter and use configured validation_split
def load_trajectories(data_paths: Union[str, List[str]],
                      max_files=100,
                      first_fraction=1.0,
                      validation_split=0.1,
                      traj_type = None,
                      random_seed=None,
                      action_repeat_counter=2,
                      mode=Mode.TRAINING):
    # --- Existing code for finding files ---
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    if random_seed is not None:
        np.random.seed(random_seed)

    all_files = []
    for data_path in data_paths:
        files = [f for f in os.listdir(data_path) if f.endswith(".json")]
        files_all_test = [f for f in os.listdir(data_path)]
        all_files.extend([(data_path, f) for f in files])
        if mode == Mode.DEBUG and len(all_files) > max_files:
            # shuffle files
            np.random.shuffle(all_files)
            all_files = all_files[:max_files * 2]

    if first_fraction < 1.0:
        # sort files by name
        all_files = sorted(all_files, key=lambda x: x[1])
        print(f"First file from all_files: {all_files[0][1]}, last file: {all_files[-1][1]}")
        all_files = all_files[:int(first_fraction * len(all_files))]
        print(f"Using only the first {first_fraction * 100:.0f}% of files.")
        print(f"First file: {all_files[0][1]}, last file: {all_files[-1][1]}")

    print(f"Found {len(all_files)} files in {len(data_paths)} data paths.")
    total_files = len(all_files)

    all_filenames = []
    for data_path, file in all_files:
        filename = os.path.join(data_path, file)
        if not os.path.isfile(filename):
             print(f"Warning: File disappeared? {filename}")
             continue

        # Apply traj_type filter if specified
        if traj_type == DatasetType.SIN_PLUS_NOISE:
            if not is_sin_plus_noise_traj(file):
                continue
        if traj_type == DatasetType.PPO:
            if not is_ppo_traj(file):
                continue
        if traj_type == DatasetType.PERIL:
            if not is_peril_traj(file):
                continue
        all_filenames.append(filename)

    # shuffle filenames
    np.random.shuffle(all_filenames)
    buffer_all = []
    validation_buffer = []
    filenames_loaded = []
    n_found = 0
    n_checked = 0
    n_found_per_path = [0] * len(data_paths)
    for filename in all_filenames:
        n_checked += 1
        # show progress every 1% of files
        if n_checked % (total_files // 100) == 0:
            print(".", end="", flush=True)
        if isinstance(max_files, int):
            if n_found >= max_files:
                break
        # check if max_files is a list
        elif isinstance(max_files, list):
            assert len(max_files) == len(data_paths), "max_files list must have the same length as data_paths"
            continue_with_next_file = False
            for i, max_files_per_path in enumerate(max_files):
                if data_paths[i] in filename:
                    if n_found_per_path[i] >= max_files_per_path:
                        continue_with_next_file = True
                        break
            if continue_with_next_file:
                print("s", end="")
                continue
        else:
            raise ValueError("max_files must be an integer or a list of integers")

        
        with open(filename, "r") as json_data:
            dict_data = json.load(json_data)
            if "next_ob" not in dict_data:
                print(f"Skipping file {filename} because 'next_ob' is missing.")
                continue
            ob = dict_data["ob"]
            next_ob = dict_data["next_ob"]
            dones = dict_data["episode_over"]
            if next_ob[-1][17] < tc[1]:
                continue
            action = dict_data["action_orig"] #[::action_repeat_counter]
            if not next_ob or not ob or not action:
                print(f"Skipping file {filename} because 'ob', 'next_ob' or 'action' is empty.")
                continue
            states = []
            next_states = []
            actions = []
            for i in list(range(0, len(ob)+1, action_repeat_counter))[:-1] + [len(ob)-1]:
                state = OrderedDict([
                    ("observation", ob[i][0:22]),
                    ("achieved_goal", ob[i][16:19]),
                    ("desired_goal", next_ob[-1][16:19]),
                ])
                next_state = OrderedDict([
                    ("observation", next_ob[i][0:22]),
                    ("achieved_goal", next_ob[i][16:19]),
                    ("desired_goal", next_ob[-1][16:19]),
                ])
                states.append(state)
                next_states.append(next_state)
                actions.append(action[i])
            
            if states[-1]['desired_goal'][2] > 0.751 or states[-1]['desired_goal'][2] < 0.749:
                continue

            traj = {
                'states': np.array(states),
                'actions': np.array(actions),
                'desired_goal': next_states[-1]['desired_goal'],
                'achieved_goal': next_states[-1]['achieved_goal'],
                'filename': filename,
                'random_traj_index': dict_data["random_traj_index"]
            }
            # print(f"Loaded {filename}")
            # print("Final achieved goal", traj['achieved_goal'], "goal in state:", ob[0][19:])
            if np.random.rand() < validation_split:
                validation_buffer.append(traj)
            else:
                buffer_all.append(traj)
            filenames_loaded.append(os.path.basename(filename))
            # copy file to new location
            # new_filename = os.path.join(data_path_new, os.path.basename(filename))
            # with open(filename, "r") as f:
            #     data = f.read()
            # with open(new_filename, "w") as f:
            #     f.write(data)
            n_found += 1
            for i, data_path in enumerate(data_paths):
                if data_path in filename:
                    n_found_per_path[i] += 1
    print(f"Loaded {n_found} trajectories.")
    show_statistics_of_traj_types(filenames_loaded, data_path)

    return buffer_all, validation_buffer


def get_state_normalizer(buffer):
    # Create and fit DataNormalizer
    states_list = []
    for traj in buffer:
        for state in traj['states']:
            states_list.append(state['observation'])
    states_array = np.array(states_list)

    # Create and fit DataNormalizer
    state_normalizer = DataNormalizer()
    state_normalizer.fit(states_array)
    return state_normalizer


def load_stats(stats_path):
    # Load from stats file
    with open(stats_path, 'r') as f:
        stats_dict = json.load(f)
    stats = {
        'obs': {
            'min': np.array(stats_dict['obs']['min'], dtype=np.float32),
            'max': np.array(stats_dict['obs']['max'], dtype=np.float32)
        },
        'goal': {
            'min': np.array(stats_dict['goal']['min'], dtype=np.float32),
            'max': np.array(stats_dict['goal']['max'], dtype=np.float32)
        },
        'action': {
            'min': np.array(stats_dict['action']['min'], dtype=np.float32),
            'max': np.array(stats_dict['action']['max'], dtype=np.float32)
        }
    }
    return stats


def load_diffusion_agent(model_path, device='cpu'):
    # Load the full checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract the model state dict and stats
    state_dict = checkpoint['model_state_dict']
    stats = checkpoint.get('stats', None)
    
    # Set dimensions as in training
    action_dim = 8  # Adjust based on your action space
    obs_dim = 22    # From LEN_OB in your constants
    goal_dim = 3    # From desired_goal dimensions
    obs_horizon = 2 # Number of past observations used during training
    action_horizon = 1 # Number of past actions used during training
    
    # Instantiate the model with EXACT same dimensions as during training
    model = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon + goal_dim,
    ).to(device)
    
    # Load the weights
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    
    return model, stats



def evaluate_diffusion_agent(env, model_path, random_ball=True, random_goal=True, ball_id=None, goal=None, n_runs=5):
    model, _ = load_diffusion_agent(model_path)
    stats = load_stats(model_path.replace('policy.pth', 'stats.json'))
    metrics_list = []
    metrics = evaluate_agent(env, model, random_ball=random_ball, random_goal=random_goal, ball_id=ball_id, goal=goal, n_runs=n_runs, diffusion_policy=True, diffusion_stats=stats)
    metrics_list.append(metrics)

    return metrics_list

    



# === Training ===
def train_agent(env, agent, buffer, validation_buffer=None, num_episodes=10, batch_size=1024, 
                learning_rate=3e-4, collect_during_training=0, specific_ball_id=None, 
                specific_goal=None, experiment_name='', output_dir=None,
                patience=100, max_grad_norm=1.0, min_improvement=0.01, sample_prob_buffer_collected_during_training = 0.5,
                n_evaluations=5, use_layer_norm=False):
    
    print("--Training for", num_episodes, "episodes", "- ball_id:", specific_ball_id, "goal:", specific_goal)

    steps_per_episode = 500
    min_len_buffer_collected_during_training = 10
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    losses = []
    metrics_list = []
    buffer_collected_during_training = []
    first_extra = True
    
    # Early stopping variables
    best_eval_reward = float('-inf')
    best_agent_state = None
    patience_counter = 0
    
    # Initial evaluation
    metrics = evaluate_agent(env,
            agent,
            random_ball=np.any(specific_ball_id==None),
            random_goal=np.any(specific_goal==None),
            ball_id=specific_ball_id,
            goal=specific_goal,
            n_runs=5)
    metrics['training_step'] = 0
    metrics_list.append(metrics)
    best_eval_reward = np.mean(metrics['rewards'])
    
    for episode in range(num_episodes):
        agent.train()
        losses_ep = []
        for step in range(steps_per_episode):
            batch_states = []
            batch_goals = []
            batch_actions = []
            
            # Batch sampling
            for _ in range(batch_size):
                if len(buffer_collected_during_training) < min_len_buffer_collected_during_training or np.random.rand() > sample_prob_buffer_collected_during_training:
                    trajectory = buffer[np.random.choice(len(buffer))]
                else:
                    trajectory = buffer_collected_during_training[np.random.choice(len(buffer_collected_during_training))]
                    if first_extra:
                        first_extra = False
                        print('----- Using extra buffer ----')

                t1 = np.random.randint(0, len(trajectory['actions']))
                s = trajectory['states'][t1]['observation']
                a = trajectory['actions'][t1]
                g = trajectory['desired_goal']
                batch_states.append(s)
                batch_goals.append(g)
                batch_actions.append(a)
                
            # Normalization
            batch_states = np.array(batch_states)
            # check if agent has attribute data_normalizer
            if hasattr(agent, 'data_normalizer') and agent.data_normalizer is not None:
                batch_states = agent.data_normalizer.transform(batch_states)

            # Convert to tensors and forward pass
            states_tensor = torch.tensor(batch_states, dtype=torch.float32)
            goals_tensor = torch.tensor(batch_goals, dtype=torch.float32)
            horizons_tensor = torch.zeros((batch_size, 1), dtype=torch.float32)
            actions_tensor = torch.tensor(batch_actions, dtype=torch.float32)
            
            
            if agent.__class__.__name__ == "NNAgentDeterministic":
                # show dimensions of inputs
                predicted_actions = agent.forward(states_tensor, goals_tensor, horizons_tensor)
                loss = nn.functional.mse_loss(predicted_actions, actions_tensor)
            else:
                predicted_mean, predicted_log_std = agent.forward(states_tensor, goals_tensor, horizons_tensor)
                std = predicted_log_std.exp()
                dist = torch.distributions.Normal(predicted_mean, std)
                nll = -dist.log_prob(actions_tensor).sum(-1)
                loss = nll.mean()
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            
            losses.append(loss.item())
            losses_ep.append(loss.item())

        # Compute validation loss and evaluation metrics
        current_val_loss = float('inf')
        if validation_buffer and len(validation_buffer) > 0:
            current_val_loss = compute_validation_loss(agent, validation_buffer, batch_size)
            print(f'Episode {episode}, Training Loss: {np.mean(losses_ep):.6f}, Validation Loss: {current_val_loss:.6f}', end=' ')
        else:
            print(f'Episode {episode}, Training Loss: {np.mean(losses_ep):.6f}', end=' ')

        # Collect additional trajectories during training if needed
        for _ in range(collect_during_training):
            set_env_to_random_ball_and_random_goal(env)
            if specific_ball_id is not None:
                env.set_ball_id(specific_ball_id)
            if specific_goal is not None:
                env.set_goal(specific_goal)
            trajectory = sample_trajectory(env, agent)
            if trajectory['achieved_goal'][1] > tc[1] - hts[1]:
                print("*", end="")
                buffer_collected_during_training.append(trajectory)

        # Evaluate agent
        metrics = evaluate_agent(env,
                               agent,
                               random_ball=np.any(specific_ball_id==None),
                               random_goal=np.any(specific_goal==None),
                               ball_id=specific_ball_id,
                               goal=specific_goal,
                               n_runs=n_evaluations)
        metrics['training_step'] = (episode + 1) * steps_per_episode
        metrics['losses'] = losses_ep
        metrics_list.append(metrics)
        
        current_eval_reward = np.mean(metrics['rewards'])
        
        # Early stopping logic
        # Check evaluation reward improvement
        if current_eval_reward > best_eval_reward * (1 + min_improvement):
            best_eval_reward = current_eval_reward
            improved = True
            # Save best model state
            best_agent_state = {k: v.cpu().clone() for k, v in agent.state_dict().items()}
        else:
            improved = False
            
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {episode + 1} episodes")
            print(f"Best evaluation reward: {best_eval_reward:.6f}")
            # Restore best model state
            if best_agent_state is not None:
                agent.load_state_dict(best_agent_state)
            break
            
        # Plotting
        if False: #episode < 3 or episode % 10 == 0:
            for i in range(1):
                plot_trajectories(env, agent, step=episode + i * 0.5, buffer=buffer,
                            dataset_traj_idx=np.random.randint(len(buffer)), 
                            ball_id=specific_ball_id, goal=specific_goal, experiment_name=experiment_name, 
                            output_dir=output_dir, plot_diff_also=True, save_json=True)

        print()
                
    return agent, metrics_list

def compute_validation_loss(agent, validation_buffer, batch_size):
    agent.eval()
    losses = []
    with torch.no_grad():
        num_batches = max(1, len(validation_buffer) // batch_size)
        for _ in range(num_batches):
            batch = [validation_buffer[np.random.choice(len(validation_buffer))] for _ in range(batch_size)]
            batch_states = []
            batch_goals = []
            batch_actions = []
            for traj in batch:
                t1 = np.random.randint(0, len(traj['actions']))
                s = traj['states'][t1]['observation']
                a = traj['actions'][t1]
                g = traj['desired_goal']
                batch_states.append(s)
                batch_goals.append(g)
                batch_actions.append(a)

            batch_states = np.array(batch_states)
            if hasattr(agent, 'data_normalizer') and agent.data_normalizer is not None:
                batch_states = agent.data_normalizer.transform(batch_states)

            states_tensor = torch.tensor(batch_states, dtype=torch.float32)
            goals_tensor = torch.tensor(batch_goals, dtype=torch.float32)
            horizons_tensor = torch.zeros((batch_size, 1), dtype=torch.float32)
            actions_tensor = torch.tensor(batch_actions, dtype=torch.float32)

            if agent.__class__.__name__ == "NNAgentDeterministic":
                predicted_actions = agent.forward(states_tensor, goals_tensor, horizons_tensor)
                loss = nn.functional.mse_loss(predicted_actions, actions_tensor)
            else:
                predicted_mean, predicted_log_std = agent.forward(states_tensor, goals_tensor, horizons_tensor)
                std = predicted_log_std.exp()
                dist = torch.distributions.Normal(predicted_mean, std)
                nll = -dist.log_prob(actions_tensor).sum(-1)
                loss = nll.mean()
            losses.append(loss.item())
    agent.train()
    return np.mean(losses)

# === Evaluation ===
def evaluate_agent(env, agent, random_ball=True, random_goal=True, ball_id=0, goal=center_goal, n_runs=5, diffusion_policy=False, diffusion_stats=None):
    if not diffusion_policy:
        agent.eval()
    all_distances = []
    all_rewards = []
    scenarios = []
    for _ in range(n_runs):
        ball_id = np.random.randint(1, 106) if random_ball else ball_id
        # sample goal randomly on the opponent side
        goal = [tc[0] - hts[0] + np.random.rand() * 2 * hts[0], 
                tc[1] - hts[1] + np.random.rand() * 2 * hts[1], 
                tc[2]] if random_goal else goal
        scenarios.append({'ball_id': ball_id, 'goal': goal})
    for scenario in scenarios:
        distances = []
        rewards = []
        # Reset environment with specific ball and goal
        env.set_ball_id(scenario['ball_id'])
        env.set_goal(scenario['goal'])
        trajectory = sample_trajectory(env, agent, greedy=True, eval=True, 
                                       diffusion_policy=diffusion_policy, diffusion_stats=diffusion_stats)
        final_distance = np.linalg.norm(trajectory['achieved_goal'] - trajectory['desired_goal'])
        all_distances.append(final_distance)
        all_rewards.append(trajectory['reward'])
    success_rates = [1 if d < 0.8 else 0 for d in all_distances]
    hit_rates = [1 if r > 0 else 0 for r in all_rewards]
    print(f"Eval: D: {np.mean(all_distances):.4f}, R: {np.mean(all_rewards):.4f}, SR: {np.mean(success_rates):.2f} HR: {np.mean(hit_rates):.2f}", end=' ')
    if not diffusion_policy:
        agent.train()
    return {'distances': all_distances, 'rewards': all_rewards, 'success_rates': success_rates, 'hit_rates': hit_rates}

def set_env_to_random_ball_and_random_goal(env):
    ball_id = np.random.randint(1, 106)
    goal = [tc[0] - hts[0] + np.random.rand() * 2 * hts[0], tc[1] - hts[1] + np.random.rand() * 2 * hts[1], tc[2]]
    env.set_ball_id(ball_id)
    env.set_goal(goal)
    return ball_id, goal

def sample_trajectory(env, agent, T=250, greedy=False, eval=False, k_step_noise = 0, noise_k_step = 0, dataset_actions = None, diffusion_policy=False, diffusion_stats=None):
    """
    Samples a trajectory using the agent in the environment.
    """
    state, _ = env.reset()
    desired_goal = state['desired_goal']
    states = []
    actions = []
    total_reward = 0
    previous_previous_observation = state['observation']
    previous_observation = state['observation']
    for t in range(T):
        states.append(state)
        if diffusion_policy:
            current_observation = state['observation']
            observation_history = np.array([previous_observation, current_observation])
            action = inference_diffusion_policy(agent, observation_history, state['desired_goal'], diffusion_stats, pred_horizon=4,
            action_horizon=1)
            action=action[0]
            previous_previous_observation = previous_observation
            previous_observation = current_observation
        else:
            action = agent.get_action(state, desired_goal, horizon=0, greedy=greedy)
        
        # if t<7:
        #     action = action * 0.0000000001
        if dataset_actions is not None:
            if t<len(dataset_actions):
                action = dataset_actions[t]
        # action = action * 0.5  # Scale down actions
        if t == noise_k_step:
            action += k_step_noise
        actions.append(action)
        state, reward, done, _, _ = env.step(action)
        print(".", end="", flush=True)
        total_reward += reward
        if done:
            # print("ep steps:", t, "reward:", reward)
            break
        
    # Use the final achieved goal as the desired goal for all states
    final_achieved_goal = state['achieved_goal']
    if not eval:
        for s in states:
            s['desired_goal'] = final_achieved_goal
        desired_goal = final_achieved_goal
    # else:
    #     print("reward:", np.round(total_reward,5), "distance:", np.round(np.linalg.norm(final_achieved_goal - desired_goal), 5), end = ' ')
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'desired_goal': desired_goal,
        'achieved_goal': final_achieved_goal,
        'reward': total_reward
    }

# === Plotting and Utilities ===
def setup_plot_style():
    """Configure matplotlib style for clean, professional plots."""
    plt.style.use('seaborn-pastel')
    
    # Custom color palette
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    
    # Font settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10
    
    # Grid settings
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    
    # Figure settings
    plt.rcParams['figure.figsize'] = (14, 8.6)
    plt.rcParams['figure.dpi'] = 150
    
    return colors

def save_plot_formats(fig, base_path, prefix='', formats=[]):
    """Save plot in multiple formats."""
    if 'png' in formats:
        fig.savefig(f'{base_path}/{prefix}.png', bbox_inches='tight', dpi=300)
    if 'tex' in formats:
        # Configure tikzplotlib settings for clean output
        tikz_settings = {
            'strict': True,
            'wrap': False,
            'axis_width': '0.8\\textwidth',
            'axis_height': '0.4\\textwidth'
        }
        tikzplotlib.save(f'{base_path}/{prefix}.tex', 
                        figure=fig,
                        **tikz_settings)
def plot_trajectories(env, agent, ball_id=-1, step=0, goal=center_goal, dataset_traj_idx=None, buffer=None, experiment_name='', output_dir=None, plot_diff_also=False, save_json=False, k_step_noise = 0, noise_k_step = 0):
    colors = setup_plot_style()
    if ball_id == -1 or ball_id is None:
        ball_id = np.random.randint(1, 106)
    env.set_ball_id(ball_id)
    if goal is None:
        goal = center_goal
    env.set_goal(goal)
    
    trajectory = sample_trajectory(env, agent, greedy=True, eval=True, k_step_noise = k_step_noise, noise_k_step = noise_k_step) # for debugging: dataset_actions = buffer[dataset_traj_idx]['actions'])
    
    # print(f"plot reward: {round(trajectory['reward'], 5)}", end=' ')

    if trajectory['reward'] >0.9999 and k_step_noise == 0:
        for noise_k_step in [0]:
            print(f"--- step: {noise_k_step} ---")
            k_step_noise = 0.00000001
            while k_step_noise<10:
                print()
                print(f"noise: {k_step_noise}", end=' ')
                plot_trajectories(env, agent, ball_id=ball_id, step = step * 1000 + k_step_noise, goal=goal, dataset_traj_idx=dataset_traj_idx, buffer=buffer, experiment_name=experiment_name, output_dir=output_dir, plot_diff_also=plot_diff_also, save_json=save_json, k_step_noise = k_step_noise, noise_k_step = noise_k_step)
                k_step_noise *= 2
            print("--- ---")


    
    for plot_diff in [False, True] if plot_diff_also else [False]:

        # print("plot_diff:", plot_diff)
        fig, axs = plt.subplots(3, 1, figsize=(14, 8.6), constrained_layout=True)

        # Plot actions
        if plot_diff:
            if dataset_traj_idx is not None:
                dataset_actions = buffer[dataset_traj_idx]['actions']
                len_min = min(len(trajectory['actions']), len(dataset_actions))
                diff_actions = np.array(trajectory['actions'][:len_min]) - np.array(dataset_actions[:len_min])
                axs[0].plot(diff_actions[:, 0], label='Difference DOF 1', color=colors[0], linewidth=2)
                axs[0].plot(diff_actions[:, 1], label='Difference DOF 2', color=colors[1], linewidth=2)
        else:
            axs[0].plot([a[0] for a in trajectory['actions']], label='Action DOF 1', 
                        color=colors[0], linewidth=2, linestyle='-')
            axs[0].plot([a[1] for a in trajectory['actions']], label='Action DOF 2', 
                        color=colors[1], linewidth=2, linestyle='-')

            if dataset_traj_idx is not None:
                dataset_actions = buffer[dataset_traj_idx]['actions']
                axs[0].plot([a[0] for a in dataset_actions], label='Dataset DOF 1', 
                        color=colors[0], linewidth=1.5, linestyle='--', alpha=0.7)
                axs[0].plot([a[1] for a in dataset_actions], label='Dataset DOF 2', 
                        color=colors[1], linewidth=1.5, linestyle='--', alpha=0.7)

        axs[0].grid(True, alpha=0.3)
        axs[0].set_ylabel('Actions')
        axs[0].legend(frameon=True, fancybox=True, framealpha=0.9)
        
        # Plot joint positions
        if plot_diff:
            if dataset_traj_idx is not None:
                dataset_joints = [s['observation'][0:4] for s in buffer[dataset_traj_idx]['states']]
                len_min = min(len(trajectory['states']), len(dataset_joints))
                diff_joints = np.array([s['observation'][0:4] for s in trajectory['states'][:len_min]]) - np.array(dataset_joints[:len_min])
                for j in range(4):
                    axs[1].plot(diff_joints[:, j], label=f'Difference Joint {j+1}', 
                            color=colors[j], linewidth=2)
        else:
            joints = [s['observation'][0:4] for s in trajectory['states']]
            dataset_joints = [s['observation'][0:4] for s in buffer[dataset_traj_idx]['states']] if dataset_traj_idx is not None else None

            for j in range(4):
                axs[1].plot([pos[j] for pos in joints], label=f'Joint {j+1}', 
                        color=colors[j], linewidth=2, linestyle='-')
                if dataset_joints:
                    axs[1].plot([pos[j] for pos in dataset_joints], label=f'Dataset Joint {j+1}', 
                            color=colors[j], linewidth=1.5, linestyle='--', alpha=0.7)

        axs[1].grid(True, alpha=0.3)
        axs[1].set_ylabel('Joint Positions')
        axs[1].legend(frameon=True, fancybox=True, framealpha=0.9)
        
        # Plot ball positions
        if plot_diff:
            if dataset_traj_idx is not None:
                dataset_ball = np.array([s['achieved_goal'] for s in buffer[dataset_traj_idx]['states']])
                len_min = min(len(trajectory['states']), len(dataset_ball)) - 1
                diff_ball = np.array([s['achieved_goal'] for s in trajectory['states'][:len_min]]) - dataset_ball[:len_min]
                for i, dim in enumerate(['X', 'Y', 'Z']):
                    axs[2].plot(diff_ball[:, i], label=f'Difference {dim}', 
                            color=colors[i], linewidth=2)
        else:
            ball_pos = np.array([s['achieved_goal'] for s in trajectory['states']])
            dataset_ball = np.array([s['achieved_goal'] for s in buffer[dataset_traj_idx]['states']]) if dataset_traj_idx is not None else None
            
            for i, dim in enumerate(['X', 'Y', 'Z']):
                axs[2].plot(ball_pos[:, i], label=f'Ball {dim}', 
                        color=colors[i], linewidth=2, linestyle='-')
                if dataset_ball is not None:
                    axs[2].plot(dataset_ball[:, i], label=f'Dataset Ball {dim}', 
                            color=colors[i], linewidth=1.5, linestyle='--', alpha=0.7)
        
        axs[2].grid(True, alpha=0.3)
        axs[2].set_ylabel('Ball Position')
        axs[2].set_xlabel('Time Step')
        axs[2].legend(frameon=True, fancybox=True, framealpha=0.9)
        
        # Save both PNG and TEX formats
        prefix = experiment_name + "_" if experiment_name else ""
        if plot_diff:
            prefix += "diff_"
        prefix += f'trajectories_{step}'

        # print("Saving plot", prefix)
        save_plot_formats(fig, output_dir, prefix)
        plt.close()

        if save_json:
            # save traj. as json
            # print(f"Saving {prefix} trajectory as JSON")
            save_trajectory(trajectory, prefix, date_in_filename=False)
    


def get_metric_name_plot(metric_name):
    if metric_name == 'rewards':
        return 'Reward'
    elif metric_name == 'distances':
        return 'Distance'
    elif metric_name == 'losses':
        return 'Loss'
    elif metric_name == 'success_rates':
        return 'Success Rate'
    elif metric_name == 'hit_rates':
        return 'Hit Rate (Reward > 0)'
    else:
        return metric_name

def plot_metrics(metrics_list, metric_name='rewards', experiment_name=''):
    colors = setup_plot_style()
    
    plt.figure(figsize=(14, 8.6))
    
    mean_metrics = []
    std_metrics = []
    training_steps = []
    
    metrics_list = sorted(metrics_list, key=lambda x: x['training_step'])
    for key, group in itertools.groupby(metrics_list, key=lambda x: x['training_step']):
        group = list(group)
        if not any(metric_name in metrics for metrics in group):
            continue
        metrics_by_metric_name = [metrics[metric_name] for metrics in group if metric_name in metrics]
        if not metrics_by_metric_name:
            continue
        
        if isinstance(metrics_by_metric_name[0], list):
            metrics_by_metric_name_flat = [item for sublist in metrics_by_metric_name for item in sublist]
        else:
            metrics_by_metric_name_flat = metrics_by_metric_name
            
        mean_metric = np.mean(metrics_by_metric_name_flat)
        std_metric = np.std(metrics_by_metric_name_flat)
        mean_metrics.append(mean_metric)
        std_metrics.append(std_metric)
        training_steps.append(key)
    
    mean_metrics = np.array(mean_metrics)
    std_metrics = np.array(std_metrics)
    
    if "loss" in metric_name:
        plt.yscale('log')
    
    plt.plot(training_steps, mean_metrics, color=colors[0], linewidth=2)
    plt.fill_between(training_steps, mean_metrics - std_metrics, mean_metrics + std_metrics,
                    color=colors[0], alpha=0.2)
    
    if metric_name == 'distances':
        plt.ylim(0, 3)
    elif metric_name == 'rewards':
        plt.ylim(-1, 1)
    elif metric_name in ['success_rates', 'hit_rates']:
        plt.ylim(0, 1)

    plt.grid(True, alpha=0.3)
    plt.xlabel('Training Steps')
    plt.ylabel(f'{get_metric_name_plot(metric_name)}')
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    
    # Adjust layout for better appearance
    plt.tight_layout()
    
    # Save both PNG and TEX formats
    prefix = f'{experiment_name}_{metric_name}'
    save_plot_formats(plt.gcf(), output_folder, prefix)
    plt.close()



def convert_ndarray(item):
    """
    Recursively converts numpy arrays/scalars, Enums, dataclasses, types,
    and other objects to JSON-serializable Python types.
    """
    if isinstance(item, np.ndarray):
        return item.tolist()
    # Handle NumPy scalars AFTER arrays
    elif isinstance(item, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(item)
    elif isinstance(item, (np.float_, np.float16, np.float32,
                          np.float64)):
        # Be careful with float precision if exact values matter
        return float(item)
    elif isinstance(item, (np.complex_, np.complex64, np.complex128)):
        return {'real': float(item.real), 'imag': float(item.imag)}
    elif isinstance(item, (np.bool_)):
        return bool(item)
    elif isinstance(item, (np.void)):
        print(f"Warning: np.void type encountered, converting to None. Value: {item}")
        return None
    # --- NEW CHECKS ---
    elif isinstance(item, Enum): # Handle Enums
        return item.value # Use the Enum's value
    elif dataclasses.is_dataclass(item) and not isinstance(item, type): # Handle dataclass instances
        # Convert dataclass to dict and recursively process its values
        return convert_ndarray(dataclasses.asdict(item))
    elif isinstance(item, type): # Handle class/type objects
        return item.__name__ # Store the class name
    elif isinstance(item, MappingProxyType): # Handle mappingproxy
         # Convert to regular dict and recurse
         # This might capture things like Class.__dict__ proxies
         print(f"Warning: Converting MappingProxyType to dict. Contents: {dict(item)}")
         return convert_ndarray(dict(item))
    # --- END NEW CHECKS ---
    elif isinstance(item, dict):
        # Recursively process standard dictionaries
        return {k: convert_ndarray(v) for k, v in item.items()}
    elif isinstance(item, (list, tuple)):
        # Recursively process lists/tuples
        return [convert_ndarray(v) for v in item]
    # --- Fallback for other objects (use cautiously) ---
    # This might catch custom objects not covered above.
    # elif hasattr(item, '__dict__') and not callable(item): # Avoid trying to serialize functions via __dict__
    #     try:
    #         # Attempt converting __dict__, but might fail or be too verbose
    #         print(f"Warning: Attempting fallback conversion for {type(item)} via __dict__")
    #         return convert_ndarray(item.__dict__)
    #     except Exception as e:
    #         print(f"Warning: Fallback __dict__ conversion failed for {type(item)}: {e}. Using str().")
    #         return str(item)
    else:
        # If it's not a known type or container, return as is (hoping it's serializable)
        # or convert to string as a last resort if errors persist.
        if isinstance(item, (int, float, str, bool, type(None))):
             return item
        else:
             # print(f"Warning: Unknown type {type(item)} encountered. Returning str representation.")
             # return str(item) # Last resort, might hide issues
             # Let's see if JSON encoder fails first, then add str() if needed
             return item # Hope it's serializable

# Make sure this updated function is used in ExperimentRunner.save_dataset_comparison_data

def save_trajectory(traj, filename, date_in_filename=True):
    """
    Saves a trajectory to a file, converting all numpy arrays to lists.
    """
    current_time = time.strftime("%Y%m%d-%H%M%S")
    if date_in_filename:
        filename = f'{filename}_{current_time}.json'
    else:
        filename = f'{filename}.json'
    filename_with_path = os.path.join(output_folder, filename)
    
    # Convert the entire trajectory to JSON-serializable format
    traj_serializable = convert_ndarray(traj)

    # Verify all numpy arrays have been converted
    def check_for_numpy(obj):
        if isinstance(obj, np.ndarray):
            raise ValueError(f"Found unconverted numpy array: {obj}")
        elif isinstance(obj, dict):
            for v in obj.values():
                check_for_numpy(v)
        elif isinstance(obj, list):
            for item in obj:
                check_for_numpy(item)
    
    try:
        check_for_numpy(traj_serializable)
    except ValueError as e:
        # print(f"Warning: {e}")
        # print("Attempting additional conversion...")
        traj_serializable = convert_ndarray(traj_serializable)
    
    with open(filename_with_path, 'w') as f:
        json.dump(traj_serializable, f)
        # print(f"Trajectory saved to {filename_with_path}")


class DatasetType(Enum):
    ALL = "all"
    PPO = "ppo"
    PERIL = "peril"
    SIN_PLUS_NOISE = "sin_plus_noise"
    

class ExperimentSetting(Enum):
    MULTI_BALL_MULTI_TRAJ_MULTI_GOAL = "multi_ball_multi_traj_multi_goal"
    MULTI_BALL_MULTI_TRAJ_MULTI_GOAL_COLLECT = "multi_ball_multi_traj_multi_goal_collect"
    MULIT_BALL_MULTI_TRAJ_SINGLE_GOAL = "multi_ball_multi_traj_single_goal"
    SINGLE_BALL_MULTI_TRAJ_MULTI_GOAL = "single_ball_multi_traj_multi_goal"
    SINGLE_BALL_SINGLE_TRAJ_MULTI_GOAL_COLLECT = "single_ball_single_traj_multi_goal_collect"
    SINGLE_BALL_SINGLE_TRAJ_SINGLE_GOAL = "single_ball_single_traj_single_goal"
    SINGLE_BALL_SINGLE_TRAJ_SINGLE_GOAL_COLLECT = "single_ball_single_traj_single_goal_collect"





@dataclass
class DatasetConfig:
    folders: List[str]  # List of folders containing trajectory data
    max_trajectories: Optional[int] = None  # Maximum number of trajectories to use (None for all)
    random_seed: Optional[int] = None  # Seed for reproducibility
    filter_type: Optional[str] = None  # Type of filtering to apply ("first_percent", "last_percent", "ball_id", "goal_radius")
    filter_value: Optional[Any] = None  # Value for filtering (percentage, ball_id, or goal position)

@dataclass
class ExperimentConfig:
    setting: ExperimentSetting
    dataset_type: DatasetType
    mode: Mode
    n_training_episodes: int
    n_evaluation_episodes: int
    n_samples: int
    collect_during_training: int
    batch_size: int
    learning_rate: float
    agent_class: Type[nn.Module]
    dataset_config: Optional[DatasetConfig] = None


class ExperimentRunner:
    def __init__(self, env, base_output_folder: str, gcsl_config: Dict):
        self.env = env
        self.base_output_folder = base_output_folder
        self.results = defaultdict(list)
        self.dataset_results = defaultdict(list)
        self.gcsl_config = gcsl_config
        
    def _filter_buffer(self, buffer: List[dict], dataset_type: DatasetType) -> List[dict]:
        if dataset_type == DatasetType.ALL:
            return buffer
            
        filename_filters = {
            DatasetType.SIN_PLUS_NOISE: is_sin_plus_noise_traj,
            DatasetType.PPO: is_ppo_traj,
            DatasetType.PERIL: is_peril_traj,
        }
        
        filter_func = filename_filters[dataset_type]
        return [traj for traj in buffer if filter_func(os.path.basename(traj["filename"]))]

    def _get_experiment_params(self, config: ExperimentConfig) -> dict:
        if config.mode == Mode.DEBUG:
            return {
                "n_training_episodes": 2,
                "n_evaluation_episodes": 2,
                "n_samples": 2,
                "batch_size": 32,
            }
        return {
            "n_training_episodes": config.n_training_episodes,
            "n_evaluation_episodes": config.n_evaluation_episodes,
            "n_samples": config.n_samples,
            "batch_size": config.batch_size,
        }

    def _setup_experiment_folder(self, config: ExperimentConfig) -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        folder_name = f"{config.setting.value}_{config.dataset_type.value}_{config.mode.name.lower()}_{timestamp}"
        folder_path = os.path.join(self.base_output_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def run_single_experiment(self, config: ExperimentConfig, buffer: List[dict]) -> Dict:
        folder_path = self._setup_experiment_folder(config)
        # Filtering happens *before* calling this function in the new structure
        # filtered_buffer = self._filter_buffer(buffer, config.dataset_type) # Remove this line
        filtered_buffer = buffer # Use the buffer passed in (already filtered)

        # Get base training params from config
        training_params_from_config = self.gcsl_config.get('training_params', {})
        agent_defaults = self.gcsl_config.get('agent_defaults', {}).get(config.agent_class.__name__, {})

        metrics_combined = []
        
        # Create a more descriptive experiment name
        dataset_desc = ""
        if config.dataset_config:
             dc = config.dataset_config
             if dc.filter_type == "first_percent":
                 dataset_desc = f"_first{dc.filter_value}pct"
             elif dc.filter_type == "last_percent":
                 dataset_desc = f"_last{dc.filter_value}pct"
             elif dc.filter_type == "ball_id":
                 dataset_desc = f"_ball{dc.filter_value}"
             elif dc.filter_type == "goal_radius":
                 dataset_desc = "_goalspecific" # Maybe add value?
            # Add identifier for which folder was used if running folder by folder
             if len(dc.folders) == 1:
                 dataset_desc += f"_folder_{os.path.basename(dc.folders[0])[:15]}" # Add part of folder name

        lr_desc = f"_lr{config.learning_rate:.1e}".replace(".", "p")
        
        experiment_name = f"{config.setting.value}_{config.dataset_type.value}{dataset_desc}{lr_desc}"
        experiment_name = experiment_name.replace('/', '_') # Sanitize name

        # Split into training and validation sets

        validation_split = 0.2
        n_validation = int(len(filtered_buffer) * validation_split)
        np.random.shuffle(filtered_buffer)
        validation_buffer = filtered_buffer[:n_validation]
        training_buffer = filtered_buffer[n_validation:]
        
        for i in range(config.n_samples):
            agent = config.agent_class(self.env, n_hidden=1024, n_layers=2)
            
            if config.setting in [ExperimentSetting.SINGLE_BALL_SINGLE_TRAJ_SINGLE_GOAL, 
                                ExperimentSetting.SINGLE_BALL_SINGLE_TRAJ_SINGLE_GOAL_COLLECT,
                                ExperimentSetting.SINGLE_BALL_SINGLE_TRAJ_MULTI_GOAL_COLLECT]:
                specific_traj = np.random.choice(training_buffer)
                training_buffer = [specific_traj]
                specific_ball_id = specific_traj["random_traj_index"]
                specific_goal = specific_traj["desired_goal"] if config.setting == ExperimentSetting.SINGLE_BALL_SINGLE_TRAJ_SINGLE_GOAL else None
            elif config.setting == ExperimentSetting.SINGLE_BALL_MULTI_TRAJ_MULTI_GOAL:
                specific_ball_id = np.random.choice([t["random_traj_index"] for t in training_buffer])
                specific_goal = None
            elif config.setting == ExperimentSetting.MULIT_BALL_MULTI_TRAJ_SINGLE_GOAL:
                specific_ball_id = None
                specific_goal = np.mean([t["desired_goal"] for t in training_buffer], axis=0)
            else:
                training_buffer = filtered_buffer
                specific_ball_id = None
                specific_goal = None
                # random trajectory for saving
                specific_traj = np.random.choice(filtered_buffer)
            
            train_agent_args = {
                "env": self.env,
                "agent": agent,
                "buffer": training_buffer,
                "validation_buffer": validation_buffer,
                "num_episodes": config.n_training_episodes,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "collect_during_training": config.collect_during_training,
                "specific_ball_id": specific_ball_id,
                "specific_goal": specific_goal,
                "experiment_name": f"{experiment_name}_sample{i}", # Make name unique per sample
                "output_dir": folder_path,
                "n_evaluations": config.n_evaluation_episodes,
                # Add parameters from gcsl_config['training_params']
                "patience": training_params_from_config.get('patience', 100),
                "max_grad_norm": training_params_from_config.get('max_grad_norm', 1.0),
                "min_improvement": training_params_from_config.get('min_improvement', 0.01),
                "sample_prob_buffer_collected_during_training": training_params_from_config.get('sample_prob_buffer_collected_during_training', 0.0)
                # Note: use_layer_norm is handled during agent creation now via agent_defaults
            }

            agent, metrics = train_agent(**train_agent_args)
            
            metrics_combined.extend(metrics)
            
            # Save trajectory samples
            if i == 0:  # Save only for first sample
                save_trajectory(specific_traj, os.path.join(folder_path, f"{experiment_name}_train_example"), date_in_filename=False)
                eval_traj = sample_trajectory(self.env, agent, greedy=True, eval=True)
                save_trajectory(eval_traj,  os.path.join(folder_path, f"{experiment_name}_eval_example"), date_in_filename=False)
        
        # Plot results
        for metric_name in ['rewards', 'distances', 'losses', 'hit_rates']:
            plot_metrics(metrics_combined, metric_name=metric_name, experiment_name=experiment_name)
            
        return {
            'metrics': metrics_combined,
            'config': config,
            'folder_path': folder_path,
            'agent_class': config.agent_class.__name__
        }

    def run_all_experiments(self, configs: List[ExperimentConfig], buffer: List[dict]):
        for config in configs:
            print("------")
            print(f"\nRunning experiment: {config.setting.value} with {config.dataset_type.value} dataset")
            print("------")
            result = self.run_single_experiment(config, buffer)
            self.results[config.setting].append(result)
            self._generate_comparison_plots()
            self._generate_agent_comparison_plots()

    def _generate_comparison_plots(self):
        # Compare settings with same dataset
        for dataset_type in DatasetType:
            relevant_results = []
            for setting in ExperimentSetting:
                setting_results = [r for r in self.results[setting] 
                                 if r['config'].dataset_type == dataset_type]
                if setting_results:
                    relevant_results.extend(setting_results)
            
            if relevant_results:
                self._plot_comparison(
                    relevant_results,
                    f"comparison_settings_{dataset_type.value}",
                    "Different Settings"
                )
        
        # Compare datasets with same setting
        for setting in ExperimentSetting:
            setting_results = self.results[setting]
            if setting_results:
                self._plot_comparison(
                    setting_results,
                    f"comparison_datasets_{setting.value}",
                    "Different Datasets"
                )

    def _plot_comparison(self, results: List[Dict], filename: str, title: str):
        colors = setup_plot_style()
        
        for metric_name in ['rewards', 'distances', 'losses', 'hit_rates']:
            plt.figure(figsize=(14, 8.6))
            
            # Group results by setting or dataset type
            is_setting_comparison = "comparison_settings_" in filename
            grouped_results = defaultdict(list)
            for result in results:
                key = result['config'].setting.value if is_setting_comparison else result['config'].dataset_type.value
                grouped_results[key].extend(result['metrics'])
            
            # Plot averaged results for each group
            for idx, (label, metrics_list) in enumerate(grouped_results.items()):
                # Get all unique training steps
                all_steps = sorted(set(m['training_step'] for m in metrics_list))
                
                # Group metrics by training step
                metrics_by_step = defaultdict(list)
                for metric in metrics_list:
                    step = metric['training_step']
                    if metric_name not in metric:
                        continue
                    if isinstance(metric[metric_name], list):
                        metrics_by_step[step].extend(metric[metric_name])
                    else:
                        metrics_by_step[step].append(metric[metric_name])
                
                # Calculate mean and std for each step
                means = [np.mean(metrics_by_step[step]) for step in all_steps]
                stds = [np.std(metrics_by_step[step]) for step in all_steps]
                
                plt.plot(all_steps, means, color=colors[idx % len(colors)], label=label)
                plt.fill_between(all_steps, 
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            color=colors[idx % len(colors)], alpha=0.2)
            
            if "loss" in metric_name:
                plt.yscale('log')

            if metric_name == 'distances':
                plt.ylim(0, 3)
            elif metric_name == 'rewards':
                plt.ylim(-1, 1)
            elif metric_name in ['success_rates', 'hit_rates']:
                plt.ylim(0, 1)
                
            plt.title(f"{title}")
            plt.xlabel("Training Steps")
            plt.ylabel(f"{get_metric_name_plot(metric_name)}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_plot_formats(plt.gcf(), self.base_output_folder, f"{filename}_{metric_name}")
            plt.close()

    def get_dataset_composition(self, buffer: List[dict]) -> Dict[str, float]:
        """Calculate the composition of different trajectory types in the buffer."""
        n_sin = 0
        n_sin_plus_noise = 0
        n_ppo = 0
        n_peril = 0
        n_total = len(buffer)
        
        for traj in buffer:
            filename = os.path.basename(traj["filename"])
            if is_sin_traj(filename):
                n_sin += 1
            if is_sin_plus_noise_traj(filename):
                n_sin_plus_noise += 1
            if is_ppo_traj(filename):
                n_ppo += 1
            if is_peril_traj(filename):
                n_peril += 1
        
        composition = {
            'sin': n_sin/n_total if n_total > 0 else 0,
            'sin_plus_noise': n_sin_plus_noise/n_total if n_total > 0 else 0,
            'ppo': n_ppo/n_total if n_total > 0 else 0,
            'peril': n_peril/n_total if n_total > 0 else 0,
        }
        return composition


    def create_dataset_label(self, group_name: str, dataset_config: Optional[DatasetConfig], buffer: List[dict]) -> str:
        """Create a descriptive label for the dataset group including size and composition."""
        # Start with the group name
        prefix = f"{group_name} - "

        # Add filter information if available and relevant
        filter_desc = ""
        if dataset_config:
            if dataset_config.filter_type == "first_percent":
                filter_desc = f"First {dataset_config.filter_value}% - "
            elif dataset_config.filter_type == "last_percent":
                 filter_desc = f"Last {dataset_config.filter_value}% - "
            elif dataset_config.filter_type == "ball_id":
                 filter_desc = f"Ball {dataset_config.filter_value} - "
            elif dataset_config.filter_type == "goal_radius":
                 goal = dataset_config.filter_value
                 if goal is not None:
                     filter_desc = f"Goal ({goal[0]:.2f},{goal[1]:.2f}) - "
            # Add other filter descriptions if needed
            # Note: 'eval_dataset' or similar logic might be implicitly handled by group_name now

        # Get basic stats
        composition = self.get_dataset_composition(buffer)
        contains_aug = any("aug" in str(traj.get("filename", "")) for traj in buffer)
        n_trajectories = len(buffer)

        # Create composition string with non-zero components
        comp_parts = []
        for traj_type, fraction in composition.items():
             if fraction > 0:
                 percentage = fraction * 100
                 comp_parts.append(f"{traj_type}: {percentage:.0f}%")

        comp_str = ", ".join(comp_parts)
        if contains_aug:
             comp_str += ", w/ aug. data"

        full_label = f"{prefix}{filter_desc}{n_trajectories} trajs ({comp_str})"
        print(f"Generated label: {full_label}")

        # Return a slightly simplified label for dictionary keys/plotting if needed,
        # or the full one if preferred. Let's use a concise one for the key.
        key_label = f"{group_name}"
        if dataset_config and dataset_config.filter_type:
             key_label += f"_{dataset_config.filter_type}"
             if dataset_config.filter_value is not None:
                  # Make value filename-safe if necessary (e.g., float goals)
                  val_str = str(dataset_config.filter_value).replace('.', 'p')
                  key_label += f"_{val_str}"

        print(f"Using key label: {key_label}")
        # return key_label # Or return full_label if you prefer that as the key
        return full_label # Let's use the descriptive label for now

    def plot_dataset_comparison(self):
        """
        Creates plots comparing results across different dataset configurations
        based on the self.dataset_results structure.
        """
        colors = setup_plot_style()

        # Metrics to plot
        metrics_to_plot = ['rewards', 'distances', 'losses', 'hit_rates']

        # Iterate through each metric
        for metric_name in metrics_to_plot:
            plt.figure(figsize=(14, 8.6))
            plot_idx = 0 # Color index

            # Iterate through each dataset group
            # Sort groups for consistent plot order
            sorted_group_names = sorted(self.dataset_results.keys())

            for group_name in sorted_group_names:
                group_variations = self.dataset_results[group_name]

                # Iterate through each variation tested within the group
                # Sort variations for consistent plot order
                sorted_variation_labels = sorted(group_variations.keys())

                for variation_label in sorted_variation_labels:
                    variation_results_list = group_variations[variation_label] # List of results (one per folder run)

                    # Aggregate metrics from all folder runs for this variation
                    all_metrics_for_variation = []
                    for result in variation_results_list:
                        all_metrics_for_variation.extend(result.get('metrics', []))

                    if not all_metrics_for_variation:
                        continue # Skip if no metrics for this variation

                    # --- Aggregate and calculate mean/std across runs and samples ---
                    # Get all unique training steps for this aggregated list
                    try:
                        all_steps = sorted(list(set(m['training_step'] for m in all_metrics_for_variation if 'training_step' in m)))
                    except: # Handle cases where metrics might be malformed
                        print(f"Warning: Could not extract training steps for {group_name}/{variation_label}. Skipping.")
                        continue


                    if not all_steps: continue # Skip if no steps found

                    metrics_by_step = defaultdict(list)
                    for metric_dict in all_metrics_for_variation:
                        step = metric_dict.get('training_step')
                        metric_value = metric_dict.get(metric_name)

                        if step is None or metric_value is None: continue

                        if isinstance(metric_value, list):
                            # If the metric itself is a list (e.g., from n_evaluations), extend
                            metrics_by_step[step].extend(metric_value)
                        else:
                            # If it's a single value (e.g., loss), append
                            metrics_by_step[step].append(metric_value)


                    # Calculate mean and std for each step
                    means = []
                    stds = []
                    steps_to_plot = []
                    
                    for step in all_steps:
                        if step in metrics_by_step and metrics_by_step[step]:
                            # Filter out potential NaNs or Infs before calculating mean/std
                            valid_metrics = [m for m in metrics_by_step[step] if np.isfinite(m)]
                            if valid_metrics:
                                means.append(np.mean(valid_metrics))
                                stds.append(np.std(valid_metrics))
                                steps_to_plot.append(step)
                            # else: skip this step if no valid metrics


                    if not steps_to_plot: continue # Skip if no valid data points

                    # --- Plotting ---
                    # Create a meaningful label: Group + Variation Info
                    # Example: "exploration - filter_all_50_LR_3e-04p"
                    plot_label = f"{group_name} - {variation_label}"

                    current_color = colors[plot_idx % len(colors)]
                    plt.plot(steps_to_plot, means,
                             label=plot_label,
                             color=current_color,
                             linewidth=2)
                    plt.fill_between(steps_to_plot,
                                     np.array(means) - np.array(stds),
                                     np.array(means) + np.array(stds),
                                     color=current_color,
                                     alpha=0.2)
                    plot_idx += 1

            # --- Finalize Plot ---
            if plot_idx == 0: # Check if anything was plotted
                 plt.close() # Close empty figure
                 print(f"Skipping plot for '{metric_name}' - no data found.")
                 continue


            if "loss" in metric_name:
                plt.yscale('log')

            # Adjust y-axis limits based on metric
            if metric_name == 'distances': plt.ylim(0, 3)
            elif metric_name == 'rewards': plt.ylim(-1, 1)
            elif metric_name in ['success_rates', 'hit_rates']: plt.ylim(0, 1)

            plt.title(f"Dataset Comparison - {get_metric_name_plot(metric_name)}")
            plt.xlabel("Training Steps")
            plt.ylabel(get_metric_name_plot(metric_name))
            # Adjust legend position for potentially many lines
            plt.legend(title="Dataset Group & Variation",
                       bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1, fontsize='small')
            plt.grid(True, alpha=0.3)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

            # Save the comparison plot
            save_plot_formats(plt.gcf(), self.base_output_folder, f"dataset_comparison_all_{metric_name}")
            plt.close()

    def save_dataset_comparison_data(self, filepath: str):
        """Saves the dataset comparison results to a JSON file."""
        print(f"\nSaving dataset comparison data to: {filepath}")
        # Use the runner's dataset_results attribute
        if not self.dataset_results:
            print("No dataset results to save.")
            return
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = convert_ndarray(self.dataset_results)

        # Verify conversion
        try:
            def check_for_numpy(obj):
                # (Copy the check_for_numpy function from save_trajectory here if needed)
                 if isinstance(obj, np.ndarray):
                     raise ValueError(f"Found unconverted numpy array: {obj}")
                 elif isinstance(obj, dict):
                     for v in obj.values():
                         check_for_numpy(v)
                 elif isinstance(obj, list):
                     for item in obj:
                         check_for_numpy(item)
            check_for_numpy(serializable_results)
        except ValueError as e:
             print(f"Warning during numpy check: {e}")
             # Attempt re-conversion just in case
             serializable_results = convert_ndarray(serializable_results)


        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4) # Use indent for readability
        print("Dataset comparison data saved successfully.")

    def run_dataset_experiments(self, configs: List[ExperimentConfig], dataset_configs: List[DatasetConfig]):
        """Runs experiments with different dataset configurations and creates comparison plots."""
        for dataset_config in dataset_configs:
            print(f"\n=== Running experiments with dataset config: {len(dataset_config.folders)} folder(s), target {dataset_config.max_trajectories} trajectories ===")
            
            
            
            # Run experiments with this dataset
            for config in configs:
                # Load data for this configuration
                dataset_buffer, _ = load_trajectories(
                    dataset_config.folders,
                    max_files=dataset_config.max_trajectories,
                    random_seed=dataset_config.random_seed,
                    traj_type=config.dataset_type,
                    action_repeat_counter=2
                )

                # show dataset composition
                dataset_label = self.create_dataset_label(dataset_config, dataset_buffer)
                print(f"Dataset composition: {dataset_label}")

                # Create a descriptive label for this dataset configuration
                dataset_label = self.create_dataset_label(dataset_config, dataset_buffer)
                print(f"Dataset composition: {dataset_label}")

                config.dataset_config = dataset_config
                print(f"\nRunning experiment: {config.setting.value} with {config.dataset_type.value} dataset")
                result = self.run_single_experiment(config, dataset_buffer)
                self.results[config.setting].append(result)
                self.dataset_results[dataset_label].append(result)
            
                # Generate comparison plots after each dataset configuration
                self._generate_comparison_plots()
                self.plot_dataset_comparison()

    def _generate_agent_comparison_plots(self):
        # Group results by setting and dataset
        grouped_results = defaultdict(lambda: defaultdict(list))
        for setting_results in self.results.values():
            for result in setting_results:
                setting_name = result['config'].setting.value
                dataset_name = result['config'].dataset_type.value
                agent_name = result['agent_class']
                grouped_results[(setting_name, dataset_name)][agent_name].append(result)
        
        # Generate plots for each (setting, dataset) combination
        for (setting_name, dataset_name), agent_results in grouped_results.items():
            for metric_name in ['rewards', 'distances', 'losses', 'hit_rates']:
                plt.figure(figsize=(14, 8.6))
                
                # Plot each agent's performance
                for agent_name, results in agent_results.items():
                    metrics_list = []
                    for result in results:
                        metrics_list.extend(result['metrics'])
                    
                    # Get all unique training steps
                    all_steps = sorted(set(m['training_step'] for m in metrics_list))
                    
                    # Group metrics by training step
                    metrics_by_step = defaultdict(list)
                    for metric in metrics_list:
                        step = metric['training_step']
                        if metric_name not in metric:
                            continue
                        if isinstance(metric[metric_name], list):
                            metrics_by_step[step].extend(metric[metric_name])
                        else:
                            metrics_by_step[step].append(metric[metric_name])
                    
                    # Calculate mean and std for each step
                    means = [np.mean(metrics_by_step[step]) for step in all_steps]
                    stds = [np.std(metrics_by_step[step]) for step in all_steps]
                    
                    # Plot
                    plt.plot(all_steps, means, label=f"{agent_name}", linewidth=2)
                    plt.fill_between(all_steps, 
                                    np.array(means) - np.array(stds),
                                    np.array(means) + np.array(stds), 
                                    alpha=0.2)
                
                # Configure plot
                # plt.title(f"Agent Comparison - {metric_name} ({setting_name}, {dataset_name})")
                plt.xlabel("Training Steps")
                plt.ylabel(get_metric_name_plot(metric_name))
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot with a unique filename
                filename = f"agent_comparison_{setting_name}_{dataset_name}_{metric_name}"
                save_plot_formats(plt.gcf(), self.base_output_folder, filename)
                plt.close()




def run_experiments(env, buffer, output_folder: str, mode: Mode = Mode.TRAINING):
    runner = ExperimentRunner(env, output_folder)
    
    base_config = {
        "n_training_episodes": 20 if mode == Mode.TRAINING else 2,
        "n_evaluation_episodes": 10 if mode == Mode.TRAINING else 2,
        "n_samples": 4 if mode == Mode.TRAINING else 2,
        "batch_size": 1024 if mode == Mode.TRAINING else 32,
        "learning_rate": 1e-2,
    }

    agent_classes = [NNAgentDeterministic, NNAgent, CBCAgent]
    
    configs = []
    for dataset_type in [DatasetType.ALL]:
        for setting in [ExperimentSetting.MULTI_BALL_MULTI_TRAJ_MULTI_GOAL]: # ExperimentSetting:
            if "single" in setting.value:
                continue 
            for agent_class in agent_classes:
                if mode == Mode.DEBUG and dataset_type not in [DatasetType.ALL]:
                    continue
                collect_during_training = 10 if "collect" in setting.value else 0
                configs.append(ExperimentConfig(
                    setting=setting,
                    dataset_type=dataset_type,
                    mode=mode,
                    collect_during_training=collect_during_training,
                    agent_class=agent_class,
                    **base_config
                ))
    
    print(f"Running {len(configs)} experiments...")
    for config in configs:
        print(f"Agent: {config.agent_class.__name__}, Setting: {config.setting.value}, Dataset: {config.dataset_type.value}")
    print("------")

    runner.run_all_experiments(configs, buffer)
    return runner

def filter_dataset(buffer, filter_type, filter_value):
    """
    Apply a filter to a dataset buffer
    
    Args:
        buffer: List of trajectory dictionaries
        filter_type: Type of filtering to apply
        filter_value: Value for filtering
        
    Returns:
        Filtered buffer
    """
    if filter_type is None or filter_value is None:
        return buffer.copy()
    
    filtered_buffer = []
    
    if filter_type == "first_percent":
        # Sort by modification time of filename (assuming it contains timestamps)
        try:
            # Try to sort by filename first (more reliable if filenames contain timestamps)
            sorted_buffer = sorted(buffer, key=lambda x: os.path.basename(x['filename']))
            print("Sorted by filename for first_percent filtering")
            if len(sorted_buffer) > 2:
                print(f"Example sort order: {os.path.basename(sorted_buffer[0]['filename'])} -> {os.path.basename(sorted_buffer[-1]['filename'])}")
        except Exception as e:
            # Fall back to modification time if sorting by filename fails
            print(f"Sorting by filename failed: {e}")
            sorted_buffer = sorted(buffer, key=lambda x: os.path.getmtime(x['filename']))
            print("Sorted by modification time for first_percent filtering")
            
        num_to_keep = max(1, int(len(sorted_buffer) * (filter_value / 100.0)))
        filtered_buffer = sorted_buffer[:num_to_keep]
        print(f"Keeping first {filter_value}% ({len(filtered_buffer)}/{len(buffer)}) files")
        
    elif filter_type == "last_percent":
        # Sort by modification time or filename
        try:
            # Try to sort by filename first (more reliable if filenames contain timestamps)
            sorted_buffer = sorted(buffer, key=lambda x: os.path.basename(x['filename']))
            print("Sorted by filename for last_percent filtering")
            if len(sorted_buffer) > 2:
                print(f"Example sort order: {os.path.basename(sorted_buffer[0]['filename'])} -> {os.path.basename(sorted_buffer[-1]['filename'])}")
        except Exception as e:
            # Fall back to modification time if sorting by filename fails
            print(f"Sorting by filename failed: {e}")
            sorted_buffer = sorted(buffer, key=lambda x: os.path.getmtime(x['filename']))
            print("Sorted by modification time for last_percent filtering")
            
        num_to_keep = max(1, int(len(sorted_buffer) * (filter_value / 100.0)))
        filtered_buffer = sorted_buffer[-num_to_keep:]
        print(f"Keeping last {filter_value}% ({len(filtered_buffer)}/{len(buffer)}) files")
        
    elif filter_type == "ball_id":
        # Filter by ball ID
        filtered_buffer = [traj for traj in buffer if traj['random_traj_index'] == filter_value]
        print(f"Keeping {len(filtered_buffer)}/{len(buffer)} files with ball ID {filter_value}")
        
    elif filter_type == "goal_radius":
        # Filter by goal proximity
        target_goal = np.array(filter_value)
        radius = 0.05  # 5cm radius
        filtered_buffer = [traj for traj in buffer if 
                          np.linalg.norm(traj['desired_goal'] - target_goal) <= radius]
        print(f"Keeping {len(filtered_buffer)}/{len(buffer)} files with goal within 5cm of {target_goal}")
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return filtered_buffer



def dataset_experiments(env, output_folder, gcsl_config: Dict, mode: Mode = Mode.TRAINING):
    """
    Run experiments comparing different dataset groups.
    Trains independently on each dataset within a group and averages results for the group.
    """
    runner = ExperimentRunner(env, output_folder, gcsl_config)

    # Get mode-specific settings
    mode_str = mode.name.lower()
    mode_settings = gcsl_config['dataset_experiment_settings']['mode_specific'][mode_str]

    base_config = {
        "n_training_episodes": mode_settings['n_training_episodes'],
        "n_evaluation_episodes": mode_settings['n_evaluation_episodes'],
        "n_samples": mode_settings['n_samples'],
        "batch_size": mode_settings['batch_size'],
    }
    max_files_for_mode = mode_settings['max_files']

    experiment_variations = []
    # Read variations from config
    learning_rates = gcsl_config['dataset_experiment_settings']['learning_rates']
    filters_to_test_json = gcsl_config['dataset_experiment_settings']['filters_to_test']
    # Convert JSON null to Python None for filters
    filters_to_test = [[f[0], f[1]] for f in filters_to_test_json]

    agent_class_name = gcsl_config['dataset_experiment_settings']['agent_class_to_use']
    # Map agent class name string to actual class
    agent_class_map = {
        "CBCAgent": CBCAgent,
        "NNAgent": NNAgent,
        "NNAgentDeterministic": NNAgentDeterministic
    }
    try:
        agent_class = agent_class_map[agent_class_name]
    except KeyError:
        raise ValueError(f"Unknown agent class name '{agent_class_name}' in gcsl_config. Valid names are: {list(agent_class_map.keys())}")

    collect_during_training = gcsl_config['dataset_experiment_settings']['collect_during_training']
    action_repeat_counter = gcsl_config['data_loading']['action_repeat_counter'] # Get from config

    # Get training defaults
    default_training_params = gcsl_config.get('training_params', {})


    for lr in learning_rates:
        for filter_type, filter_value in filters_to_test:
            # Create a unique label for this variation (used as inner key)
            filter_type_str = filter_type if filter_type is not None else "all"
            filter_value_str = filter_value if filter_value is not None else ""
            variation_label = f"filter_{filter_type_str}_{filter_value_str}_LR_{lr:.0e}"
            variation_label = variation_label.replace(".","p") # Make label cleaner

            # Include agent defaults if needed, but train_agent takes them separately for now
            agent_default_hps = gcsl_config.get('agent_defaults', {}).get(agent_class_name, {})

            experiment_variations.append({
                "variation_label": variation_label,
                "learning_rate": lr,
                "filter_type": filter_type,
                "filter_value": filter_value,
                "agent_class": agent_class,
                "setting": ExperimentSetting.MULTI_BALL_MULTI_TRAJ_MULTI_GOAL,
                "dataset_type": DatasetType.ALL,
                "collect_during_training": collect_during_training,
                "mode": mode,
                "collect_during_training": collect_during_training,
                **default_training_params,
            })

    # --- Structure to store results: Dict[group_name, Dict[variation_label, List[result]]] ---
    runner.dataset_results = defaultdict(lambda: defaultdict(list))
    dataset_folder_groups_from_config = gcsl_config['paths']['dataset_folder_groups']

    # Iterate through each defined dataset group from config
    for group_name, group_folders in dataset_folder_groups_from_config.items():
        print(f"\n{'='*10} Processing Dataset Group: {group_name} ({len(group_folders)} folders) {'='*10}")

        # --- Loop through each dataset FOLDER within the group ---
        for folder_path in group_folders:
            print(f"\n  --- Processing Folder: {os.path.basename(folder_path)} ---")

            # Load trajectories ONLY from the current folder
            try:
                dataset_buffer, _ = load_trajectories(
                    [folder_path], # Load from this single folder
                    max_files=max_files_for_mode,
                    first_fraction=1.0, # Load all files from this folder
                    validation_split=gcsl_config['data_loading']['validation_split'],
                    random_seed=42,
                    action_repeat_counter=action_repeat_counter,
                    mode=mode,
                )
                print(f"  Loaded {len(dataset_buffer)} trajectories from {os.path.basename(folder_path)}")
            except FileNotFoundError:
                print(f"  Warning: Folder not found: {folder_path}. Skipping.")
                continue

            if not dataset_buffer:
                print(f"  Warning: No trajectories loaded from {os.path.basename(folder_path)}. Skipping.")
                continue

            # Apply experiment variations using data from this specific folder
            for variation_params_template in experiment_variations:
                current_config_params = deepcopy(variation_params_template)
                variation_label = current_config_params.pop("variation_label") # Get the label

                # Create DatasetConfig specific to this folder and filter
                dataset_config = DatasetConfig(
                    folders=[folder_path], # Reflects the single folder used
                    max_trajectories=None,
                    random_seed=42,
                    filter_type=current_config_params.pop("filter_type"),
                    filter_value=current_config_params.pop("filter_value")
                )

                # Filter the buffer loaded from the single dataset
                filtered_buffer = filter_dataset(
                    dataset_buffer,
                    dataset_config.filter_type,
                    dataset_config.filter_value
                )

                if not filtered_buffer:
                    print(f"    Warning: Buffer empty after filter ({dataset_config.filter_type}, {dataset_config.filter_value}). Skipping variation.")
                    continue

                print(f"\n    --- Running Variation '{variation_label}' on {os.path.basename(folder_path)} ---")
                print(f"    Using {len(filtered_buffer)} trajectories after filtering.")

                # Create the final ExperimentConfig for run_single_experiment
                # Combine base mode settings with variation specifics
                exp_config_dict = {**base_config, **current_config_params}
                # Ensure agent_class is set correctly
                exp_config_dict['agent_class'] = variation_params_template['agent_class']
                # Add dataset_config
                exp_config_dict['dataset_config'] = dataset_config

                # Remove potential duplicates or unnecessary keys before creating ExperimentConfig
                keys_to_remove = ['filter_type', 'filter_value'] # These are in dataset_config now
                for key in keys_to_remove:
                    exp_config_dict.pop(key, None)

                # Pop extra training params that are handled by train_agent directly
                extra_training_params_for_train_agent = {}
                for key in list(default_training_params.keys()):
                     if key in exp_config_dict:
                         extra_training_params_for_train_agent[key] = exp_config_dict.pop(key)


                try:
                    # Pass only the arguments expected by ExperimentConfig constructor
                    exp_config = ExperimentConfig(
                        setting=exp_config_dict.pop('setting'),
                        dataset_type=exp_config_dict.pop('dataset_type'),
                        mode=exp_config_dict.pop('mode'),
                        n_training_episodes=exp_config_dict.pop('n_training_episodes'),
                        n_evaluation_episodes=exp_config_dict.pop('n_evaluation_episodes'),
                        n_samples=exp_config_dict.pop('n_samples'),
                        collect_during_training=exp_config_dict.pop('collect_during_training'),
                        batch_size=exp_config_dict.pop('batch_size'),
                        learning_rate=exp_config_dict.pop('learning_rate'),
                        agent_class=exp_config_dict.pop('agent_class'),
                        dataset_config=exp_config_dict.pop('dataset_config')
                    )
                except KeyError as e:
                     print(f"Error creating ExperimentConfig: Missing key {e}")
                     print(f"Current params: {exp_config_dict}")
                     continue # Skip this variation


                # Run the single experiment configuration
                # Note: run_single_experiment creates its own output subfolder based on timestamp etc.
                result = runner.run_single_experiment(exp_config, filtered_buffer)

                # Add metadata for grouping later
                result['group_name'] = group_name
                result['folder_path'] = folder_path
                result['variation_label'] = variation_label
                result['learning_rate'] = exp_config.learning_rate # Store for convenience
                result['dataset_size_filtered'] = len(filtered_buffer)

                # Store results under the group and variation label
                # The list associated with [group_name][variation_label] will grow
                # as we process more folders within the same group for the same variation.
                runner.dataset_results[group_name][variation_label].append(result)

                # Optional: Generate intermediate comparison plots (within-run plots)
                # runner._generate_comparison_plots()
                # runner._generate_agent_comparison_plots()

        output_json_path = os.path.join(output_folder, "dataset_comparison_data.json")
        runner.save_dataset_comparison_data(output_json_path)

    print("\n===== Dataset Group Experiments Completed =====")
    print(f"Comparison data saved to: {output_json_path}")
    print("Run the separate plotting script to generate comparison plots.")
    




@dataclass
class HPConfig:
    """Hyperparameter configuration"""
    name: str
    values: List[Any]

class HPSearchRunner:
    def __init__(self, env, base_output_folder: str):
        self.env = env
        self.base_output_folder = base_output_folder
        self.results = {}
        
    def run_hp_search(self, 
                     parameter_configs: List[HPConfig],
                     base_params: Dict[str, Any],
                     buffer: List[dict],
                     n_seeds: int = 3):
        """Run hyperparameter search experiments"""
        
        for param_config in parameter_configs:
            print(f"\n=== Testing {param_config.name} ===")
            self.results[param_config.name] = []
            
            for value in param_config.values:
                print(f"\nTrying {param_config.name} = {value}")
                metrics_combined = []
                
                # Run multiple seeds
                for seed in range(n_seeds):
                    print(f"Seed {seed + 1}/{n_seeds}")
                    
                    # Create new parameter dictionary with current value
                    current_params = deepcopy(base_params)
                    
                    # Handle nested parameters (e.g., 'network.n_hidden')
                    if '.' in param_config.name:
                        parts = param_config.name.split('.')
                        if parts[0] not in current_params:
                            current_params[parts[0]] = {}
                        current_params[parts[0]][parts[1]] = value
                    else:
                        current_params[param_config.name] = value

                    if current_params['sample_prob_buffer_collected_during_training'] > 0:
                        current_params['collect_during_training'] = 10

                    if current_params['collect_during_training'] > 0:
                        current_params['sample_prob_buffer_collected_during_training'] = 0.5
                    
                    
                    # State normalilser
                    state_normalizer = None
                    if 'data_normalization' in current_params:
                        data_normalization = current_params.pop('data_normalization')
                        if data_normalization:
                            state_normalizer = get_state_normalizer(buffer)

                    use_layer_norm = current_params.pop('use_layer_norm', False)

                    # Create and train agent
                    if 'network' in current_params:
                        net_params = current_params.pop('network')
                        print("net_params: ", net_params)
                        agent = CBCAgent(self.env, data_normalizer=state_normalizer, use_layer_norm=use_layer_norm, **net_params)
                    else:
                        agent = CBCAgent(self.env, data_normalizer=state_normalizer, use_layer_norm=use_layer_norm, n_hidden=1024, n_layers=2)
                    
                    # Split data into training and validation
                    validation_split = 0.2
                    n_validation = int(len(buffer) * validation_split)
                    np.random.shuffle(buffer)
                    validation_buffer = buffer[:n_validation]
                    training_buffer = buffer[n_validation:]
                    
                    if "max_buffer_size" in current_params:
                        max_buffer_size = current_params.pop("max_buffer_size")
                    else:
                        max_buffer_size = len(training_buffer)

                    # Train agent
                    _, metrics = train_agent(
                        self.env,
                        agent,
                        training_buffer[:max_buffer_size],
                        validation_buffer=validation_buffer,
                        n_evaluations=20,
                        **current_params
                    )
                    metrics_combined.extend(metrics)
                
                self.results[param_config.name].append({
                    'value': value,
                    'metrics': metrics_combined
                })
                
            # Plot results for this parameter
            self._plot_parameter_comparison(param_config.name)
    
    def _plot_parameter_comparison(self, param_name: str):
        """Create comparison plot for a parameter's different values"""
        plt.figure(figsize=(14, 8.6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.results[param_name])))
        
        for idx, result in enumerate(self.results[param_name]):
            value = result['value']
            metrics_list = result['metrics']
            
            # Get all unique training steps
            all_steps = sorted(set(m['training_step'] for m in metrics_list))
            
            # Group metrics by training step
            rewards_by_step = defaultdict(list)
            for metric in metrics_list:
                step = metric['training_step']
                if isinstance(metric['rewards'], list):
                    rewards_by_step[step].extend(metric['rewards'])
                else:
                    rewards_by_step[step].append(metric['rewards'])
            
            # Calculate mean and std for each step
            means = [np.mean(rewards_by_step[step]) for step in all_steps]
            stds = [np.std(rewards_by_step[step]) for step in all_steps]
            
            plt.plot(all_steps, means, 
                    label=f'{param_name}={value}',
                    color=colors[idx],
                    linewidth=2)
            plt.fill_between(all_steps,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           color=colors[idx],
                           alpha=0.2)
        
        plt.title(f'Impact of {param_name} on Training Performance')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.ylim(-1, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        # plt.savefig(f'{self.base_output_folder}/hp_search_{param_name}.png', 
        #            bbox_inches='tight', dpi=300)
        save_plot_formats(plt.gcf(), self.base_output_folder, f"hp_search_{param_name}")
        plt.close()

def run_hp_search(env, buffer, output_folder: str):
    """Run comprehensive hyperparameter search"""
    
    # Create output folder for HP search
    hp_search_folder = os.path.join(output_folder, "hp_search")
    os.makedirs(hp_search_folder, exist_ok=True)

    # Base parameters
    base_params = {
        "num_episodes": 12,
        "batch_size": 2048,
        "learning_rate": 1e-3,
        "collect_during_training": 0,
        "sample_prob_buffer_collected_during_training": 0.0,
        "max_grad_norm": 1.0,
        "patience": 100,
        "min_improvement": 0.01,
        "network": {
            "n_hidden": 1024,
            "n_layers": 1,
            "dropout": 0.0
        },
        "output_dir": hp_search_folder 
    }
    
    # Define parameter configurations to test
    hp_configs = [
        HPConfig("max_buffer_size", [2000, 200]),
        HPConfig("learning_rate", [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2]),
        HPConfig("use_layer_norm", [True, False]),
        HPConfig("data_normalization", [True, False]),
        HPConfig("sample_prob_buffer_collected_during_training", [0.0, 0.1, 0.2, 0.5, 0.7, 0.9]),
        HPConfig("network.dropout", [0.0, 0.1, 0.2, 0.3, 0.5, 0.9]),
        HPConfig("network.n_layers", [1, 2, 3, 4, 5]),
        HPConfig("batch_size", [16, 64, 256, 1024, 4096]),
        HPConfig("max_grad_norm", [0.1, 0.2, 0.5, 1.0, 20.0]),
        HPConfig("network.n_hidden", [32, 128, 512, 2048]),
        HPConfig("num_episodes", [2, 5, 10]),
        HPConfig("collect_during_training", [0, 5, 10, 20]),
        HPConfig("patience", [1, 2, 3, 4, 5]),
        HPConfig("min_improvement", [0.005, 0.01, 0.02, 0.05])
    ]
    
    # Create runner and execute search
    runner = HPSearchRunner(env, output_folder)
    runner.run_hp_search(hp_configs, base_params, buffer, n_seeds=3)
    
    return runner



def one_robot_traj_one_goal_one_ball_test(env, buffer, output_folder):
    random_traj = np.random.choice(buffer)
    ball_idx = random_traj["random_traj_index"]
    goal = random_traj["desired_goal"]
    agent = NNAgentDeterministic(env, n_hidden=2048, n_layers=1)
    agent, metrics = train_agent(env, agent, [random_traj], num_episodes=30, n_evaluations=1, specific_ball_id=ball_idx, specific_goal=goal, output_dir=output_folder, learning_rate=1e-5, patience=200, batch_size=256)
    
    # plot results
    for metric_name in ['rewards', 'distances', 'losses', 'hit_rates']:
        plot_metrics(metrics, metric_name=metric_name, experiment_name='single_ball_single_traj_single_goal')





def main():
    # Create environment
    print("--- Creating environment... ---", flush=True)
    env = create_environment()
    print("--- Environment initialized ---", flush=True)

    # Check for debug/video flag
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    video_mode = "--video" in sys.argv or "-v" in sys.argv
    mode = Mode.DEBUG if debug_mode else Mode.VIDEO if video_mode else Mode.TRAINING
    print(f"Running in {mode.name} mode", flush=True)

    # Get base output folder from config
    base_output_folder_config = gcsl_config['paths']['output_folder']

    # Create timestamp-based output folder for this specific run *inside* the base folder
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    mode_suffix = mode.name.lower()
    run_output_folder = os.path.join(base_output_folder_config, f"run_{mode_suffix}_{timestamp}")

    if not os.path.exists(run_output_folder):
        os.makedirs(run_output_folder)
    print(f"Results for this run will be saved in: {run_output_folder}")

    # # Pass the loaded config to dataset_experiments
    dataset_experiments(env, run_output_folder, gcsl_config, mode)

    # evaluate diffusion policy
    # model_path = "/path/to/policy.pth"
    # evaluate_diffusion_agent(env, model_path, random_ball=True, random_goal=True, ball_id=None, goal=None, n_runs=5)


if __name__ == '__main__':
    main()
