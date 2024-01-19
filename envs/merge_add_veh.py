import traci
import gym
import numpy as np
import os
import random
from utils.utils import *


def qew_merge_add_veh(t,mainlane_demand,merge_lane_demand):
    hdv_main_='hdvmain_'
    hdv_merge_ = 'hdvmerge_' 
    cav_main_ = 'cavmain_' ## this is incase we want to control RL vehicles
    cav_merge_ = 'cavmerge_' 
    hdv_main=hdv_main_+str(t)
    hdv_merge=hdv_merge_+str(t)

    departspeed = random.choice([27, 30])

    # Mainline demand
    if 0 <= t <= 54000:
        inflow_rate_mainline = mainlane_demand /36000
    elif 54000 < t <= 90000:
        inflow_rate_mainline = max(0, mainlane_demand * (1 - (t - 54000) / 36000))/36000
    else:
        inflow_rate_mainline = 0
    # print('inflow_rate_mainline',inflow_rate_mainline)
    
    # Sample from a uniform distribution for mainline
    u_mainline = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for mainline
    if u_mainline < inflow_rate_mainline:
        traci.vehicle.add(hdv_main, routeID='route_2', typeID='human', departLane='random', departSpeed=departspeed)
        # traci.vehicle.setSpeed(veh_name, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name, 3)

    # Merge lane demand
    if 0 <= t <= 18000:
        inflow_rate_merge_lane = merge_lane_demand *(t/18000)/36000
    elif 18000 < t <= 54000:
        inflow_rate_merge_lane = merge_lane_demand/36000
    elif 54000 < t <= 72000:
        inflow_rate_merge_lane = max(0, merge_lane_demand * (1 - (t - 54000) / 18000))/36000
    else:
        inflow_rate_merge_lane = 0

    # print('inflow_rate_mergeline',inflow_rate_merge_lane)
    # Sample from a uniform distribution for merge lane
    u_merge_lane = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for merge lane
    if u_merge_lane < inflow_rate_merge_lane:
        traci.vehicle.add(hdv_merge, routeID='route_1', typeID='human', departLane='free', departSpeed=departspeed)
        # traci.vehicle.setSpeed(veh_name4, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name4,3)

    veh_id_list=list(traci.vehicle.getIDList())

    return veh_id_list



def syn_merge_add_veh(t,mainlane_demand,merge_lane_demand):
    hdv_main_='hdvmain_'
    hdv_merge_ = 'hdvmerge_' 
    cav_main_ = 'cavmain_' 
    cav_merge_ = 'cavmerge_' 
    hdv_main=hdv_main_+str(t)
    hdv_merge=hdv_merge_+str(t)

    departspeed = random.choice([27, 30])

    # Mainline demand
    if 0 <= t <= 2*5400:
        inflow_rate_mainline = mainlane_demand /36000
    elif 2*5400 < t <= 2*9000:
        inflow_rate_mainline = max(0, mainlane_demand * (1 - (t - 54000) / 36000))/36000
    else:
        inflow_rate_mainline = 0
    # print('inflow_rate_mainline',inflow_rate_mainline)
    
    # Sample from a uniform distribution for mainline
    u_mainline = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for mainline
    if u_mainline < inflow_rate_mainline:
        traci.vehicle.add(hdv_main, routeID='route_0', typeID='human', departLane='random', departSpeed=departspeed)
        traci.vehicle.setSpeed(hdv_main, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name, 3)

    # Merge lane demand
    if 0 <= t <= 2*1800:
        inflow_rate_merge_lane = merge_lane_demand *(t/18000)/36000
    elif 2*18000 < t <= 2*54000:
        inflow_rate_merge_lane = merge_lane_demand/36000
    elif 2*540 < t <= 2*720:
        inflow_rate_merge_lane = max(0, merge_lane_demand * (1 - (t - 54000) / 18000))/36000
    else:
        inflow_rate_merge_lane = 0

    # print('inflow_rate_mergeline',inflow_rate_merge_lane)
    # Sample from a uniform distribution for merge lane
    u_merge_lane = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for merge lane
    if u_merge_lane < inflow_rate_merge_lane:
        traci.vehicle.add(hdv_merge, routeID='route_1', typeID='human', departLane='free', departSpeed=departspeed)
        traci.vehicle.setSpeed(hdv_merge, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name4,3)

    veh_id_list=list(traci.vehicle.getIDList())

    return veh_id_list

