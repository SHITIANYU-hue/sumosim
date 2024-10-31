import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import traci
import os
from envs.add_veh import syn_merge_add_veh_constant
from argparse import ArgumentParser
import sumolib
from utils.utils import *

# 配置参数
parser = ArgumentParser('parameters')
parser.add_argument("--horizon", type=int, default=50000, help='number of simulation steps, (default: 6000)')
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--exp_tag', type=str, default="default_exp", help="Experiment tag for saving checkpoint files")
parser.add_argument('--use_random_demand', type=bool, default=True, help="Whether to use random demand values or fixed values (default: True)")


args = parser.parse_args()
outID_250=['250_0loop','250_1loop']
outID_246=['246_1loop']
outID_319=['319_0loop','319_1loop']
outID_248= ['248_1loop'] ##248_1loop

class CustomSUMOEnv:
    def __init__(self, gui=False, warm_up_step=100,state_dim=4, 
                 network_conf="networks/merge_synth_small/sumoconfig.sumo2.cfg",  ## 可能需要修改为sim=1
                 network_xml="networks/merge_synth_small/merge.net2.xml"):
        self.gui = gui
        self.network_conf = network_conf
        self.network_xml = network_xml
        self.warm_up_step = warm_up_step
        self.state_dim = state_dim
        self.rl_vehicles_state = {}
    
    def start(self, gui=False):
        # 该方法启动 SUMO 仿真
        self.gui = gui
        self.net = sumolib.net.readNet(self.network_xml)
        self.curr_step = 0
        self.collision = False
        self.done = False
        self.stats_path = 'results_small/stat1.xml'
        
        # 设置 SUMO 二进制文件路径
        home = os.getenv("HOME")
        sumoBinary = home + "/code/sumo/bin/sumo-gui" if self.gui else home + "/code/sumo/bin/sumo"
        
        # 配置 SUMO 启动命令
        sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
        save_stats = True
        if save_stats:
            sumoCmd.extend(["--statistic-output", self.stats_path, "--duration-log.statistics"])
        
        # 启动 SUMO
        traci.start(sumoCmd)
        self.lane_ids = traci.lane.getIDList()
        
        # 预热步骤
        for step in range(self.warm_up_step):
            traci.simulationStep()
    
    def reset(self, gui=False):
        # 重置环境：重新启动仿真并初始化状态
        self.start(gui)
        veh_id_list = list(traci.vehicle.getIDList())
        self.rl_vehicles_state = {veh_id: self.get_state(veh_id) for veh_id in veh_id_list}
        return np.zeros(self.state_dim)  # 返回初始状态

    def get_state(self, veh_id):
        # 定义获取车辆状态的方法
        # 假设您已经定义了获取单个车辆状态的逻辑
        # 这里返回一个状态数组（具体实现根据您的需求来调整）
        pass

# 定义Actor和Critic网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 定义DDPG智能体
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = deque(maxlen=10000)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        action=self.actor(state).cpu().data.numpy().flatten()
        return action

    
    def train(self, batch_size=64, discount=0.99, tau=0.005):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state))
        action = torch.FloatTensor(np.array(action))
        reward = torch.FloatTensor(np.array(reward))
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(np.array(done))

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1 - done) * discount * target_q).detach()

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def add_to_replay_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def save_checkpoint(self, filepath):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)

# 获取状态信息
def get_state():
    inflow_319 = traci.edge.getLastStepVehicleNumber("319")
    inflow_246 = traci.edge.getLastStepVehicleNumber("246")

    inflow_248 = traci.edge.getLastStepVehicleNumber("248")
    outflow_250 = traci.edge.getLastStepVehicleNumber("250")
    occupancy_276 = traci.lane.getLastStepOccupancy("276_0")
    return np.array([inflow_319, inflow_248, inflow_246,outflow_250, occupancy_276])

# 设置速度限制
def set_speed_limit(action,speed_limit=25):
    action=np.absolute(action)
    # print('action',action)
    traci.edge.setMaxSpeed("319", speed_limit*action[0])
    traci.edge.setMaxSpeed("246", speed_limit*action[1])
    traci.edge.setMaxSpeed("994", speed_limit*action[2])
    traci.edge.setMaxSpeed("248", speed_limit*action[3])

# 设置时间头间距（tau）
def set_time_headway(action, min_tau=0.1):
    action = np.absolute(action)
    edges = ["319", "246", "994", "248"]
    for edge, act in zip(edges, action):
        vehicle_ids = traci.edge.getLastStepVehicleIDs(edge)
        for veh_id in vehicle_ids:
            tau_value = min_tau * (1 + act)  # 根据action调整tau值
            traci.vehicle.setTau(veh_id, tau_value)

# 计算奖励
def calculate_reward():
    outflow = calc_outflow(outID_250)/2000
    rampflow = calc_outflow(outID_248)/2000
    collisions = len(traci.simulation.getCollidingVehiclesIDList())
    # print('rampflow',rampflow,'outflow',outflow,'collision',collisions)
    reward = rampflow + outflow - 100*collisions
    return reward

# 初始化自定义环境和智能体
state_dim = 5
action_dim = 4
max_action = 1.0
agent = DDPGAgent(state_dim, action_dim, max_action)
env = CustomSUMOEnv(gui=args.render,state_dim=state_dim)

# 配置流量需求
mainlane_demands = [165, 330, 440, 495, 550]
merge_lane_demands = [150, 90]
mainlane_demand = random.choice(mainlane_demands)
merge_lane_demand = random.choice(merge_lane_demands)
control_type='headway' ## or speed
# 训练循环
episodes = 200
checkpoint_dir = "/home/tianyu/code/uoftexp/sumosim/video_data/1031"
warmup=1000
for episode in range(episodes):
    state = env.reset(gui=args.render)  # 每个 episode 重置环境
    episode_reward = 0
    t = 1

    while t < args.horizon:
        # print('t',t)
        # 每过10000步，随机选择新的流量需求
        if t % 10000 == 0:
            if args.use_random_demand:
                mainlane_demand = random.choice(mainlane_demands)
                merge_lane_demand = random.choice(merge_lane_demands)
            else:
                mainlane_demand = 550
                merge_lane_demand = 150
            print(f"Step {t}: New mainlane demand: {mainlane_demand}, New merge lane demand: {merge_lane_demand}")
        
        veh_id_list = syn_merge_add_veh_constant(t, mainlane_demand, merge_lane_demand)  # 添加车辆
        traci.simulationStep()  # 前进仿真
        if t > warmup:
            action = agent.select_action(state)
            if control_type=='speed':
                set_speed_limit(action)
            if control_type=='headway':
                set_time_headway(action)

            next_state = get_state()
            # print('state',next_state)
            reward = calculate_reward()
            # print('reward',reward)
            done = False  # 定义结束条件

            agent.add_to_replay_buffer(state, action, reward, next_state, done)
            agent.train(batch_size=64)

            state = next_state
            episode_reward += reward

            if done:
                break
        t += 1

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

    ## 每隔10个episode保存一次模型检查点
    if (episode + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"{args.exp_tag}_checkpoint_episode_{episode + 1}.pth")
        agent.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved at episode {episode + 1}")

    traci.close()