from configparser import ConfigParser
from argparse import ArgumentParser
import traci
import numpy as np
import os
from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG
import random
import time
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

args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser,args.algo)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'
exp_tag='pporingplatoon'
unix_timestamp = int(time.time())

env = sumo_env_ring()

action_dim = 1
state_dim = 4
state_rms = RunningMeanStd(state_dim)

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(f'score/{exp_tag}_{args.algo}_{unix_timestamp}')
else:
    writer = None
if args.algo == 'ppo' :
    agent = PPO(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'sac' :
    agent = SAC(writer, device, state_dim, action_dim, agent_args)
elif args.algo == 'ddpg' :
    from utils.noise import OUNoise
    noise = OUNoise(action_dim,0)
    agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)

if args.load != 'no':
    print('load',args.load)
    agent.load_state_dict(torch.load("./model_weights/"+args.load))
# for i in range(len(flow_rates)):
state = env.reset(gui=args.render)
action={}




simdur = args.horizon  # assuming args.horizon represents the total simulation duration

data=[]

t=1
controller_='secrmrl' # or idm
veh_id_list=syn_ring_add_veh()
gaps=[]
speeds=[]
actions=[]
while t < simdur:
    veh_id_list = [veh_id for veh_id in traci.vehicle.getIDList() if not veh_id.startswith('lead_')]
    print('id list',len(veh_id_list))

    lane = 0
    acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
    gaps_list=[]
    vels_list=[]
    action_list=[]
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
        if controller_=='secrm':
            controller=secrmController()
            action_=controller.get_accel(state[veh_id_list[i]])
            action[veh_id_list[i]] = [0,action_]
        if controller_=='secrmrl':
            controller=secrmController()
            mu,sigma = agent.get_action(torch.from_numpy(state[veh_id_list[i]]).float().to(device))
            dist = torch.distributions.Normal(mu,sigma[0])
            action_s = dist.sample()
            action_secrm=controller.get_accel(state[veh_id_list[i]])
            action_=action_s.cpu().detach().numpy()
            action[veh_id_list[i]] = [0,action_secrm+action_[0]]
        action_list.append(action_)
    print('actions',action)
    if len(veh_id_list)==3:
        gaps.append(gaps_list)
        speeds.append(vels_list)
        actions.append(action_list)

    next_state_, reward_info, done, info = env.step(
        action, veh_id_list)
    if done:
        # print('rl vehicle run out of network!!')
        pass
    state=next_state_
    t=t+1

np.save(f'results/speed_{controller_}.npy',speeds)
np.save(f'results/gaps_{controller_}.npy',gaps)
np.save(f'results/actions_{controller_}.npy',actions)

env.close()