from configparser import ConfigParser
from argparse import ArgumentParser
import traci
import numpy as np
import os
import random
from envs.synthetic_small_env_merge import sumo_env_merge
from envs.merge_add_veh import syn_merge_add_veh
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



    
env = sumo_env_merge()



outID_250=['250_0loop','250_1loop']
outID_246=['246_0loop','246_1loop']
outID_319=['319_0loop','319_1loop']
outID_994=['994_0loop']


# state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)


# for i in range(len(flow_rates)):
state = env.reset(gui=args.render)
action={}



mainlane_demand = 3300 ##5500, 4400,3850,3300
merge_lane_demand = 0
interval = 2000  # interval for calculating average statistics
simdur = args.horizon  # assuming args.horizon represents the total simulation duration
curflow = 0
curflow_994 = 0
curflow_246 = 0
curflow_319 = 0

curdensity = 0
avg_speeds=[]
cos,hcs,noxs,pmxs=[],[],[],[]
inflows = []
inflows_994 = []
inflows_246 = []
inflows_319 = []
time_step=0.1
total_travel_time=0
densities = []
data=[]

t=1

while t < simdur:
    print('step', t)
    lane = 0
    acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
    veh_id_list=syn_merge_add_veh(t,mainlane_demand,merge_lane_demand)
    vehPermid = get_vehicle_number('276_0') + get_vehicle_number('276_1') + get_vehicle_number('276_2')
    vehPerout = get_vehicle_number('250_0') + get_vehicle_number('250_1') 

    # veh_number_total=traci.vehicle.getIDCount()+int(len(traci.simulation.getPendingVehicles())) ##i find that it will make it run very slow
    # veh_number_total=traci.vehicle.getIDCount()+len(traci.edge.getPendingVehicles('994')) ## this is to count waiting vehicle in the merge
    veh_number_total=traci.vehicle.getIDCount()

    total_travel_time = total_travel_time+ (time_step*veh_number_total)/3600



    curdensity += vehPermid / traci.lane.getLength('276_0')
    # print('curflow',curflow,'cudensity',curdensity)

    if t % interval == 0:
        # append average flow and density for the last interval
        curflow = curflow + calc_outflow(outID_250)
        curflow_994 = curflow_994 + calc_outflow(outID_994)
        curflow_246 = curflow_246 + calc_outflow(outID_246)
        curflow_319 = curflow_319 + calc_outflow(outID_319)
        avg_speed=(get_meanspeed('276_0')+get_meanspeed('276_1')+get_meanspeed('276_2'))/3 ### this is only bottlneck's speed

        co,hc,nox,pmx,all_avg_speed=calc_emission_speed(lanelist=['276_0','276_1','276_2','246_0','246_1','248_0']) ## i consider to calculate all edges emission and avg speed (for whole network)
        cos.append(co),hcs.append(hc),noxs.append(nox),pmxs.append(pmx)
        inflows.append(curflow )
        inflows_994.append(curflow_994 )
        inflows_246.append(curflow_246 )
        inflows_319.append(curflow_319 )
        avg_speeds.append(all_avg_speed)

        densities.append(curdensity / interval)

        # reset averages
        curflow = 0
        curflow_994=0
        curflow_246=0
        curflow_319=0
        curdensity = 0

    t = t + 1

    for i in range(len(veh_id_list)):
        action[veh_id_list[i]] = [1, 0]
    
    next_state_, reward_info, done, info = env.step(
        action, veh_id_list)

    
    if done:
        # print('rl vehicle run out of network!!')
        pass

    
    # # # # Save the average values
    # np.save(f'results/250/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows)
    # np.save(f'results/994/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_994)
    # np.save(f'results/246/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_246)
    # np.save(f'results/319/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_319)
    # np.save(f'results/Main{mainlane_demand}_merge{merge_lane_demand}average_density2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', densities)
print('total travel time (h):',total_travel_time)
print('average bottleneck speed:',np.mean(avg_speeds))
print('average emission: ','CO:',np.mean(cos),'HC:',np.mean(hcs),'NOX:',np.mean(noxs),'PMX:',np.mean(pmxs))
env.close()