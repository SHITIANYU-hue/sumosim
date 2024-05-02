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



def syn_merge_add_veh(t,mainlane_demand,merge_lane_demand,factor=0.5):
    hdv_main_='hdvmain_'
    hdv_merge_ = 'hdvmerge_' 
    cav_main_ = 'cavmain_' 
    cav_merge_ = 'cavmerge_' 
    hdv_main=hdv_main_+str(t)
    hdv_merge=hdv_merge_+str(t)

    departspeed = random.choice([27, 30])
    # speed_mode=32
    # Mainline demand
    if 0 <= t <= factor*54000:
        inflow_rate_mainline = mainlane_demand /36000
    elif factor*54000 < t <= factor*90000:
        inflow_rate_mainline = max(0, mainlane_demand * (1 - (t - 54000) / 36000))/36000
    else:
        inflow_rate_mainline = 0
    # print('inflow_rate_mainline',inflow_rate_mainline)
    
    # Sample from a uniform distribution for mainline
    u_mainline = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for mainline
    if u_mainline < inflow_rate_mainline:
        traci.vehicle.add(hdv_main, routeID='route_0', typeID='rl', departLane='random', departSpeed=departspeed)
        traci.vehicle.setSpeed(hdv_main, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name, 3)
        traci.vehicle.setLaneChangeMode(hdv_main,0b001000000000)

    # Merge lane demand
    if 0 <= t <= factor*18000:
        inflow_rate_merge_lane = merge_lane_demand /36000
    elif factor*18000 < t <= factor*54000:
        inflow_rate_merge_lane = merge_lane_demand/36000
    elif factor*54000 < t <= factor*72000:
        inflow_rate_merge_lane = max(0, merge_lane_demand * (1 - (t - 54000) / 18000))/36000
    else:
        inflow_rate_merge_lane = 0
    # print('inflow_rate_mergeline',inflow_rate_merge_lane)
    # Sample from a uniform distribution for merge lane
    u_merge_lane = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for merge lane
    if u_merge_lane < inflow_rate_merge_lane:
        traci.vehicle.add(hdv_merge, routeID='route_1', typeID='rl', departLane='free', departSpeed=departspeed)
        traci.vehicle.setSpeed(hdv_merge, departspeed)
        # traci.vehicle.setSpeedFactor(veh_name4,3)
        traci.vehicle.setLaneChangeMode(hdv_merge,0b001000000000)

    veh_id_list=list(traci.vehicle.getIDList())

    return veh_id_list

def syn_merge_add_veh_constant(t,mainlane_demand,merge_lane_demand,factor=0.5):
    hdv_main_='hdvmain_'
    hdv_merge_ = 'hdvmerge_' 
    cav_main_ = 'cavmain_' 
    cav_merge_ = 'cavmerge_' 
    hdv_main=hdv_main_+str(t)
    hdv_merge=hdv_merge_+str(t)

    departspeed = random.choice([27, 30])
    # speed_mode=32
    # Mainline demand
    inflow_rate_mainline = mainlane_demand /3600

    # print('inflow_rate_mainline',inflow_rate_mainline)
    
    # Sample from a uniform distribution for mainline
    u_mainline = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for mainline
    if u_mainline < inflow_rate_mainline:
        traci.vehicle.add(hdv_main, routeID='route_0', typeID='rl', departLane='random', departSpeed=departspeed)
        traci.vehicle.setColor(hdv_main,[0,255,255,255])
        traci.vehicle.setSpeed(hdv_main, departspeed)
        
        # traci.vehicle.setSpeedFactor(veh_name, 3)
        # traci.vehicle.setLaneChangeMode(hdv_main,0b001000000000)

    # Merge lane demand
    inflow_rate_merge_lane = merge_lane_demand /3600

    # print('inflow_rate_mergeline',inflow_rate_merge_lane)
    # Sample from a uniform distribution for merge lane
    u_merge_lane = np.random.uniform(0, 1)
    # Check if a vehicle should be generated based on the sampled value for merge lane
    if u_merge_lane < inflow_rate_merge_lane:
        traci.vehicle.add(hdv_merge, routeID='route_1', typeID='rl', departLane='free', departSpeed=departspeed)
        traci.vehicle.setSpeed(hdv_merge, departspeed)
        traci.vehicle.setColor(hdv_merge,[255,0,0,255])
        # traci.vehicle.setSpeedFactor(veh_name4,3)
        # traci.vehicle.setLaneChangeMode(hdv_merge,0b001000000000)

    veh_id_list=list(traci.vehicle.getIDList())

    return veh_id_list

def syn_ring_add_veh(departspeed_lead=25,departspeed_follow=10,speed_limit=25,num_follow=3,departLane=0):
    leader='lead_'
    follower = 'follower_' 
    traci.vehicle.add(leader, routeID='route_0', typeID='human', departLane=departLane, departSpeed=departspeed_lead)
    traci.vehicle.setLaneChangeMode(leader,0b001000000000)
    speed_mode=32 ## disable all check for speed

    for i in range(num_follow):
        follower_=follower+str(i)
        traci.vehicle.add(follower_, routeID='route_0', typeID='rl', departLane=departLane, departSpeed=departspeed_follow)
        traci.vehicle.setLaneChangeMode(follower_,0b001000000000)
        traci.vehicle.setSpeedMode(follower_, speed_mode)

    veh_id_list=list(traci.vehicle.getIDList())
    return veh_id_list
