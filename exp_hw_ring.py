from configparser import ConfigParser
from argparse import ArgumentParser
import traci
import numpy as np
import os
import random
from envs.synthetic_ring_env import sumo_env_ring
import json
from envs.add_veh import syn_ring_add_veh
from agents.controller import *
from utils.utils import *
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default ='gym_sumo_qew_env_merge')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--manual', type=bool, default=False, help="(default: False)")
parser.add_argument('--horizon', type=int, default=1800, help='number of simulation steps, (default: 6000)')
parser.add_argument('--coop', type=float, default=0, help='cooperative factor for human vehicles')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
args = parser.parse_args()



    
env = sumo_env_ring()



# for i in range(len(flow_rates)):
state = env.reset(gui=args.render)
action={}




simdur = args.horizon  # assuming args.horizon represents the total simulation duration

data=[]

t=1
controller_='idm' # or idm
veh_id_list=syn_ring_add_veh()
gaps=[]
speeds=[]
while t < simdur:
    veh_id_list = [veh_id for veh_id in traci.vehicle.getIDList() if not veh_id.startswith('lead_')]
    print('id list',len(veh_id_list))

    lane = 0
    acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
    gaps_list=[]
    vels_list=[]
    
    info_lists = [[] for _ in range(len(veh_id_list))]

    for i in range(len(veh_id_list)):
        gaps_=state[veh_id_list[i]][2]
        gaps_list.append(gaps_)
        vels_=state[veh_id_list[i]][0]
        vels_list.append(vels_)
        if controller_=='idm':
            controller=IDMController()
            action_=controller.get_accel(state[veh_id_list[i]])
            action[veh_id_list[i]] = [0,action_]
        if controller_=='gipps':
            controller=GippsController()
            action_=controller.get_accel(state[veh_id_list[i]])
            action[veh_id_list[i]] = [0,action_]
    print('actions',action)
    if len(veh_id_list)==3:
        gaps.append(gaps_list)
        speeds.append(vels_list)

    next_state_, reward_info, done, info = env.step(
        action, veh_id_list)
    if done:
        # print('rl vehicle run out of network!!')
        pass
    state=next_state_
    t=t+1

np.save(f'results/speed_{controller_}.npy',speeds)
np.save(f'results/gaps_{controller_}.npy',gaps)

env.close()