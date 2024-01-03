from configparser import ConfigParser
from argparse import ArgumentParser
import traci
import gym
import numpy as np
import os
import random
from envs.sumo_qew_env_merge import sumo_qew_env_merge
from envs.qew_merge_add_veh import qew_merge_add_veh
import json

from utils.utils import *
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default ='gym_sumo_qew_env_merge')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--manual', type=bool, default=False, help="(default: False)")
parser.add_argument('--horizon', type=int, default=180000, help='number of simulation steps, (default: 6000)')
parser.add_argument('--coop', type=float, default=0, help='cooperative factor for human vehicles')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
args = parser.parse_args()



    
env = sumo_qew_env_merge()



outID_9728=['9728_0loop','9728_1loop','9728_2loop']
outID_9832=['9832_0loop','9832_1loop','9832_2loop']
outID_9575=['9575_0loop','9575_1loop','9575_2loop']
outID_9813=['9813_0loop']


# state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)


# for i in range(len(flow_rates)):
state = env.reset(gui=args.render)
action={}



mainlane_demand = 3300 ##5500, 4400,3850,3300
merge_lane_demand = 1250
interval = 2000  # interval for calculating average statistics
simdur = args.horizon  # assuming args.horizon represents the total simulation duration
curflow = 0
curflow_9813 = 0
curflow_9832 = 0
curflow_9575 = 0

curdensity = 0
avg_speeds=[]
cos,hcs,noxs,pmxs=[],[],[],[]
inflows = []
inflows_9813 = []
inflows_9832 = []
inflows_9575 = []
time_step=0.1
total_travel_time=0
densities = []
data=[]

t=1

while t < simdur:
    print('step', t)
    lane = 0
    acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
    veh_id_list=qew_merge_add_veh(t,mainlane_demand,merge_lane_demand)
    vehPermid = get_vehicle_number('9712_1') + get_vehicle_number('9712_2') + get_vehicle_number('9712_3')
    vehPerout = get_vehicle_number('9728_0') + get_vehicle_number('9728_1') + get_vehicle_number('9728_2')

    veh_number = get_vehicle_number('9712_0')+get_vehicle_number('9712_1') + get_vehicle_number('9712_2') + get_vehicle_number('9712_3')
    # veh_number_total=traci.vehicle.getIDCount()+int(len(traci.simulation.getPendingVehicles())) ##i find that it will make it run very slow
    # veh_number_total=traci.vehicle.getIDCount()+len(traci.edge.getPendingVehicles('9813')) ## this is to count waiting vehicle in the merge
    veh_number_total=traci.vehicle.getIDCount()

    total_travel_time = total_travel_time+ (time_step*veh_number_total)/3600



    curdensity += vehPermid / traci.lane.getLength('9712_1')
    # print('curflow',curflow,'cudensity',curdensity)

    if t % interval == 0:
        # append average flow and density for the last interval
        curflow = curflow + calc_outflow(outID_9728)
        curflow_9813 = curflow_9813 + calc_outflow(outID_9813)
        curflow_9832 = curflow_9832 + calc_outflow(outID_9832)
        curflow_9575 = curflow_9575 + calc_outflow(outID_9575)
        avg_speed=(get_meanspeed('9712_1')+get_meanspeed('9712_2')+get_meanspeed('9712_3')+get_meanspeed('9712_0'))/4 ### this is only bottlneck's speed

        co,hc,nox,pmx,all_avg_speed=calc_emission_speed() ## i consider to calculate all edges emission and avg speed (for whole network)
        cos.append(co),hcs.append(hc),noxs.append(nox),pmxs.append(pmx)
        inflows.append(curflow )
        inflows_9813.append(curflow_9813 )
        inflows_9832.append(curflow_9832 )
        inflows_9575.append(curflow_9575 )
        avg_speeds.append(all_avg_speed)
        # print('flow stas','9728',calc_outflow(outID_9728),'9832',calc_outflow(outID_9832),'9575',calc_outflow(outID_9575))

        densities.append(curdensity / interval)
        # print('average laneflow:', curflow / interval, 'average density', curdensity / interval,'average speed',np.mean(avg_speeds))

        # reset averages
        curflow = 0
        curflow_9813=0
        curflow_9832=0
        curflow_9575=0
        curdensity = 0

    t = t + 1

    for i in range(len(veh_id_list)):
        action[veh_id_list[i]] = [2, 0]
    
    next_state_, reward_info, done, info = env.step(
        action, veh_id_list,sumo_lc=False, sumo_carfollow=False, stop_and_go=False, car_follow='Gipps', lane_change='SECRM')

    
    if done:
        # print('rl vehicle run out of network!!')
        pass

    
    # # # # Save the average values
    np.save(f'results/9728/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows)
    np.save(f'results/9813/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_9813)
    np.save(f'results/9832/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_9832)
    np.save(f'results/9575/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_9575)
    np.save(f'results/Main{mainlane_demand}_merge{merge_lane_demand}average_density2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', densities)
print('total travel time (h):',total_travel_time)
print('average bottleneck speed:',np.mean(avg_speeds))
print('average emission: ','CO:',np.mean(cos),'HC:',np.mean(hcs),'NOX:',np.mean(noxs),'PMX:',np.mean(pmxs))
env.close()
