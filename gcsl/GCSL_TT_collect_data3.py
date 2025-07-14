import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import gym
import os
import sys
import time
import csv

from learning_table_tennis_from_scratch.hysr_one_ball_env import HysrOneBallEnv
from learning_table_tennis_from_scratch.hysr_many_ball_env import HysrManyBallEnv
from learning_table_tennis_from_scratch.hysr_goal_env import HysrGoalEnv
from learning_table_tennis_from_scratch.rl_config import RLConfig
from learning_table_tennis_from_scratch.rl_config import OpenAIRLConfig
from learning_table_tennis_from_scratch.hysr_one_ball import HysrOneBallConfig

import json
from collections import OrderedDict
import copy
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from functools import partial
import random


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

with open(gcsl_config_path, 'r') as f:
    gcsl_config = json.load(f, object_pairs_hook=OrderedDict)

reward_config_file = main_config.get("reward_config")
hysr_one_ball_config_file = main_config.get("hysr_config")
data_path = gcsl_config['paths']['data_path']
log_file_path = gcsl_config['paths']['data_path'] + '/log.csv'
vsgcsl_path = gcsl_config['paths']['vsgcsl_path']
number_steps = gcsl_config['augmentation_params']['number_steps']



env_tt = HysrManyBallEnv(reward_config_file=reward_config_file, hysr_one_ball_config_file=hysr_one_ball_config_file, log_episodes=True)
print("--- HysrManyBallEnv ---")

all_files = [f for f in os.listdir(data_path) if ".json" in f]

def sample_filename():
    return data_path + np.random.choice(all_files)

class RecomputeStateConfig:
    def __init__(self):
        self.epsilon = np.array([0.0, 0.0, 0.0])
        self.rotation_matrix_contactee = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.vel_plus = np.zeros(3)
        self.mirror_y = False

def get_robot1_recompute_config():
    config = RecomputeStateConfig()
    config.epsilon = np.array([0.78, 0.78, 0.78])
    config.rotation_matrix_contactee = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    config.vel_plus.fill(0.0)
    config.mirror_y = True
    return config

def load_data(filename):
    with open(filename, "r") as json_data:
        dict_data = json.load(json_data)
        ball_pos = [x[16:19] for x in dict_data["ob"]]
        ball_vel = [x[19:22] for x in dict_data["ob"]]
        rob_pos = [x[0] for x in dict_data["fk"]]
        rob_vel = [x[1] for x in dict_data["fk"]]
        racket_pos = [x[2] for x in dict_data["fk"]]
        racket_vel = [x[3] for x in dict_data["fk"]]
        racket_ori = [x[4] for x in dict_data["fk"]]
        timestamp = [x[5] for x in dict_data["fk"]]
        action = [x for x in dict_data["action_orig"]]
        reward = sum([x for x in dict_data["reward"]])
        ball_traj_idx = dict_data["random_traj_index"]
    return np.array(ball_pos), np.array(ball_vel), np.array(rob_pos), np.array(rob_vel), np.array(racket_pos), np.array(racket_vel), np.array(racket_ori), np.array(timestamp), np.array(action), reward, ball_traj_idx

def load_final_ball_pos(filename):
    with open(filename, "r") as json_data:
        dict_data = json.load(json_data)
        final_ball_pos = dict_data["next_ob"][-1][16:19]
        return np.array(final_ball_pos)

def check_data(filename):
    try:
        with open(filename, "r") as json_data:
            dict_data = json.load(json_data)
            return "next_ob" in dict_data.keys()
    except:
        return False

def load_ball_id(filename):
    with open(filename, "r") as json_data:
        dict_data = json.load(json_data)
        ball_traj_idx = dict_data["random_traj_index"]
        return ball_traj_idx

class ContactState:
    def __init__(self, ball_position, ball_velocity, contactee_position, contactee_velocity, contactee_orientation, time_stamp):
        self.ball_position = np.array(ball_position)
        self.ball_velocity = np.array(ball_velocity)
        self.contactee_position = np.array(contactee_position)
        self.contactee_velocity = np.array(contactee_velocity)
        self.contactee_orientation = np.array(contactee_orientation)
        self.time_stamp = time_stamp/(10**9)

def load_contact_states(filename, until_contact=False):
    ball_pos, ball_vel, _, _, racket_pos, racket_vel, racket_ori, timestamps, _, _, _ = load_data(filename)
    contact_states = []
    for i in range(len(ball_pos)):
        state = ContactState(
            ball_position=ball_pos[i],
            ball_velocity=ball_vel[i],
            contactee_position=racket_pos[i],
            contactee_velocity=racket_vel[i],
            contactee_orientation=racket_ori[i],
            time_stamp=timestamps[i]
        )
        contact_states.append(state)
    return contact_states

def log_ball_transform(filename, target_position, ball_id, success, cause_of_failure, time, error, expected_landing_point, final_position):
    if not os.path.isfile(log_file_path):
        with open(log_file_path, mode='w') as log_file:
            log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            log_writer.writerow(["filename", "target_position", "ball_id", "success", "cause_of_failure", "time", "error", "expected_landing_point", "final_position"])
    with open(log_file_path, mode='a') as log_file:
        log_writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        log_writer.writerow([filename, target_position, ball_id, success, cause_of_failure, time, error, expected_landing_point, final_position])

def simulate_2(states, env_tt, filename, ball_id=None, action_repeat_counter=1):
    ball_pos, ball_vel, _, _, racket_pos, racket_vel, _, _, action, reward_orig, idx_ball_traj = load_data(filename)
    if ball_id is not None:
        env_tt.set_ball_id(ball_id, extra_balls=True)
    
    ball_pos_env_list = []
    ball_vel_env_list = []
    
    obs, _ = env_tt.reset()
    if env_tt.__class__.__name__ == "HysrGoalEnv":
        ball_pos_env = obs['observation'][16:19]
        ball_vel_env = obs['observation'][19:22]
    elif env_tt.__class__.__name__ == "HysrManyBallEnv":
        ball_pos_env = obs[16:19]
        ball_vel_env = obs[19:22]
    else:
        raise ValueError("Unknown environment type: {}".format(env_tt.__class__.__name__))

    ball_pos_env_list.append(ball_pos_env)
    ball_vel_env_list.append(ball_vel_env)
    done = False
    idx = 0
    
    while not done:
        action_idx = idx * action_repeat_counter
        if action_idx > len(action)-1:
            action_idx = np.random.randint(0, len(action)-1)
        obs, reward, done, _, info = env_tt.step(action[action_idx])
        
        if env_tt.__class__.__name__ == "HysrGoalEnv":
            ball_pos_env = obs['observation'][16:19]
            ball_vel_env = obs['observation'][19:22]
        elif env_tt.__class__.__name__ == "HysrManyBallEnv":
            ball_pos_env = obs[16:19]
            ball_vel_env = obs[19:22]
        else:
            raise ValueError("Unknown environment type: {}".format(env_tt.__class__.__name__))

        if idx<len(action)-1:
            idx += 1
        ball_pos_env_list.append(ball_pos_env)
        ball_vel_env_list.append(ball_vel_env)

    final_position = ball_pos_env_list[-1]
    return reward, final_position, ball_pos_env_list, ball_vel_env_list

def replace_ball_and_dump(filename, new_filename, states, ball_traj_idx=-1, final_ball_pos=None):
    with open(filename, "r") as json_data:
        dict_data = json.load(json_data)
        for i in range(len(dict_data["ob"])):
            if i<len(states):
                dict_data["ob"][i][16:19] = states[i].ball_position.tolist()
                dict_data["ob"][i][19:22] = states[i].ball_velocity.tolist()
                if i<len(states)-1:
                    dict_data["next_ob"][i][16:19] = states[i+1].ball_position.tolist()
                    dict_data["next_ob"][i][19:22] = states[i+1].ball_velocity.tolist()
                else:
                    dict_data["next_ob"][i][16:19] = final_ball_pos.tolist()
                    dict_data["next_ob"][i][19:22] = np.array([0.0, 0.0, 0.0]).tolist()
            else:
                del dict_data["ob"][-1]
                del dict_data["action_orig"][-1]
                del dict_data["action_casted"][-1]
                del dict_data["prdes"][-1]
                del dict_data["reward"][-1]
        dict_data["random_traj_index"] = ball_traj_idx
        with open(new_filename, "w") as json_new:
            json.dump(dict_data, json_new)

action_repeat_counter = 2

def create_augmented_data(filename):
    config = get_robot1_recompute_config()
    states = load_contact_states(filename, until_contact=False)
    landing_position = load_final_ball_pos(filename)
    if landing_position[1]<0.5:
        return None
    _, final_position, _, _ = simulate_2(states, env_tt, filename, ball_id=None, action_repeat_counter=action_repeat_counter)
    return final_position

np.random.seed(int(time.time()))
i=0
while i<number_steps:
    filename = sample_filename()
    result = create_augmented_data(filename)
    if result is None:
        print("x", end="", flush=True)
    else:
        i += 1

print("\n--- Augmentation done ---", flush=True)

sys.exit(0)


    
    