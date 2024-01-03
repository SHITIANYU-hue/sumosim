import numpy as np
import torch
import traci
from math import atan2, degrees
from scipy.spatial import distance

class Dict(dict):
    def __init__(self,config,section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)
    def __getattr__(self,val):
        return self[val]
    
def make_transition(state,action,reward,next_state,done,log_prob=None):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['log_prob'] = log_prob
    transition['done'] = done
    return transition

def make_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    for i in range(full_batch_size // mini_batch_size):
        indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
        yield [x[indices] for x in value[1:]]
        
def convert_to_tensor(*value):
    device = value[0]
    return [torch.tensor(x).float().to(device) for x in value[1:]]

class ReplayBuffer():
    def __init__(self, action_prob_exist, max_size, state_dim, num_action):
        self.max_size = max_size
        self.data_idx = 0
        self.action_prob_exist = action_prob_exist
        self.data = {}
        
        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        if self.action_prob_exist :
            self.data['log_prob'] = np.zeros((self.max_size, 1))
    def put_data(self, transition):
        idx = self.data_idx % self.max_size
        self.data['state'][idx] = transition['state']
        self.data['action'][idx] = transition['action']
        self.data['reward'][idx] = transition['reward']
        self.data['next_state'][idx] = transition['next_state']
        self.data['done'][idx] = float(transition['done'])
        if self.action_prob_exist :
            self.data['log_prob'][idx] = transition['log_prob']
        
        self.data_idx += 1
    def sample(self, shuffle, batch_size = None):
        if shuffle :
            sample_num = min(self.max_size, self.data_idx)
            rand_idx = np.random.choice(sample_num, batch_size,replace=False)
            sampled_data = {}
            sampled_data['state'] = self.data['state'][rand_idx]
            sampled_data['action'] = self.data['action'][rand_idx]
            sampled_data['reward'] = self.data['reward'][rand_idx]
            sampled_data['next_state'] = self.data['next_state'][rand_idx]
            sampled_data['done'] = self.data['done'][rand_idx]
            if self.action_prob_exist :
                sampled_data['log_prob'] = self.data['log_prob'][rand_idx]
            return sampled_data
        else:
            return self.data
    def size(self):
        return min(self.max_size, self.data_idx)
    
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def calc_bottlespeed(bottleneck_detector):
    speed = []
    for detector in bottleneck_detector:
        dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
        if dspeed < 0:
            dspeed = 0                                              
            #The value of no-vehicle signal will affect the value of the reward
        speed.append(dspeed)
    return np.mean(np.array(speed))

def calc_emission_speed():
    vidlist = traci.edge.getIDList()
    lanelist=['9575_0','9575_1','9575_2','9712_0','9712_1','9712_2','9712_3','9813_0']
    co = []
    hc = []
    nox = []
    pmx = []
    avg_speed=[]
    for vid in vidlist:
        co.append(traci.edge.getCOEmission(vid))
        hc.append(traci.edge.getHCEmission(vid))
        nox.append(traci.edge.getNOxEmission(vid))
        pmx.append(traci.edge.getPMxEmission(vid))
    for lane in lanelist:
        avg_speed.append(traci.lane.getLastStepMeanSpeed(lane))
    return np.sum(np.array(co)),np.sum(np.array(hc)),np.sum(np.array(nox)),np.sum(np.array(pmx)),np.mean(avg_speed)


def set_vsl(v,VSLlist):
    number_of_lane = len(VSLlist)
    for j in range(number_of_lane):
        traci.lane.setMaxSpeed(VSLlist[j], v[j])

# check lane change and speed mode params: https://github.com/flow-project/flow/blob/master/flow/core/params.py

def angle_between(p1, p2, rl_angle):
	xDiff = p2[0] - p1[0]
	yDiff = p2[1] - p1[1]
	angle = degrees(atan2(yDiff, xDiff))
	# Adding the rotation angle of the agent
	angle += rl_angle
	angle = angle % 360
	return angle


def get_distance(a, b):
	return distance.euclidean(a, b)

def map_action(value,clamp=3):
	output_value = (value + clamp) / clamp
	return round(output_value)

def calc_outflow(outID): 
    state = []
    statef = []
    for detector in outID:
        veh_num = traci.inductionloop.getIntervalVehicleNumber(detector)
        state.append(veh_num)
    return np.sum(np.array(state))


get_vehicle_number = lambda lane_id: traci.lane.getLastStepVehicleNumber(lane_id)
# calculate flow sum up the vehicle speeds on the lane and divide by the length of the lane to obtain flow in veh/s
get_lane_flow = lambda lane_id: (traci.lane.getLastStepMeanSpeed(lane_id) * traci.lane.getLastStepVehicleNumber(lane_id))/traci.lane.getLength(lane_id)
get_meanspeed = lambda lane_id: traci.lane.getLastStepMeanSpeed(lane_id)