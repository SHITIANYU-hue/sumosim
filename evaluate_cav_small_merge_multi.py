from configparser import ConfigParser
from argparse import ArgumentParser
from agents.controller import *
import traci
import numpy as np
import os
import random
from envs.synthetic_small_env_merge import sumo_env_merge
from envs.add_veh import syn_merge_add_veh_constant
import json

from utils.utils import *
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default ='gym_sumo_qew_env_merge')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--manual', type=bool, default=False, help="(default: False)")
parser.add_argument('--horizon', type=int, default=18000, help='number of simulation steps, (default: 6000)')
parser.add_argument('--p', type=float, default=0, help='politeness factor for human vehicles')
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
outID_248= ['248_0loop']
outID_994=['994_0loop']

outID_276_0=['276_0loop','276_1loop','276_2loop']
outID_276_1=['276_3loop','276_4loop','276_5loop']
outID_276_2=['276_6loop','276_7loop','276_8loop']
outID_276_3=['276_9loop','276_10loop','276_11loop']
outID_276_4=['276_12loop','276_13loop','276_14loop']




# state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)


# for i in range(len(flow_rates)):
state = env.reset(gui=args.render)
action={}

controller='coopsecrmruletest'

mainlane_demand =  3300 #0.6*5500 ##5500, 4400,3850,3300
merge_lane_demand = 900 #0.6* 1500 #1000
interval = 200  # interval for calculating average statistics
simdur = args.horizon  # assuming args.horizon represents the total simulation duration
curflow = 0
curflow_994 = 0
curflow_246 = 0
curflow_319 = 0
curflow_248 = 0
curflow_276_0=0
curflow_276_1=0
curflow_276_2=0
curflow_276_3=0
curflow_276_4=0

v_276_0s=[]
v_276_1s=[]
v_276_2s=[]
v_276_3s=[]
v_276_4s=[]



curdensity = 0
avg_speeds=[]
cos,hcs,noxs,pmxs=[],[],[],[]
inflows = []
inflows_994 = []
inflows_246 = []
inflows_319 = []
inflows_248 = []

time_step=0.1
total_travel_time=0
densities = []
data=[]

t=1
agent=coopsecrmController(p=args.p)
while t < simdur:
    if t%10000==0:
        print('step', t)
    lane = 0
    acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
    veh_id_list=syn_merge_add_veh_constant(t,mainlane_demand,merge_lane_demand)
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
        curflow_248 = curflow_248 + calc_outflow(outID_248)

        curflow_276_0 = curflow_276_0 + calc_outflow(outID_276_0)
        curflow_276_1 = curflow_276_1 + calc_outflow(outID_276_1)
        curflow_276_2 = curflow_276_2 + calc_outflow(outID_276_2)
        curflow_276_3 = curflow_276_3 + calc_outflow(outID_276_3)
        curflow_276_4 = curflow_276_4 + calc_outflow(outID_276_4)

        v_276_0=calc_instanspeed(outID_276_0)
        v_276_1=calc_instanspeed(outID_276_1)
        v_276_2=calc_instanspeed(outID_276_2)
        v_276_3=calc_instanspeed(outID_276_3)
        v_276_4=calc_instanspeed(outID_276_4)

        v_276_0s.append(v_276_0)
        v_276_1s.append(v_276_1)
        v_276_2s.append(v_276_2)
        v_276_3s.append(v_276_3)
        v_276_4s.append(v_276_4)




        avg_speed=(get_meanspeed('276_0')+get_meanspeed('276_1')+get_meanspeed('276_2'))/3 ### this is only bottlneck's speed

        co,hc,nox,pmx,avg_speed=calc_emission_speed(lanelist=['276_0','276_1','276_2','246_0','246_1','248_0']) ## i consider to calculate all edges emission and avg speed (for whole network)
        cos.append(co),hcs.append(hc),noxs.append(nox),pmxs.append(pmx)

        inflows.append(curflow )
        inflows_994.append(curflow_994 )
        inflows_246.append(curflow_246 )
        inflows_319.append(curflow_319 )
        inflows_248.append(curflow_248 )
        avg_speeds.append(avg_speed)

        densities.append(curdensity / interval)

        # reset averages
        curflow = 0
        curflow_994=0
        curflow_246=0
        curflow_319=0
        curflow_248=0
        curdensity = 0

    t = t + 1

    for i in range(len(veh_id_list)):
        action[veh_id_list[i]] = agent.get_accel(veh_id_list[i])

    _, next_state_, reward_info, done, info = env.step(
        action, veh_id_list)

    # if done:
    #     # print('rl vehicle run out of network!!')
    #     pass
    # print('avg speed',avg_speeds)
    # # # Save the average values

    np.save(f'results_small_new/250/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', inflows)

    np.save(f'results_small_new/994/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', inflows_994)
    np.save(f'results_small_new/248/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', inflows_248)
    np.save(f'results_small_new/246/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', inflows_246)
    np.save(f'results_small_new/319/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', inflows_319)

    np.save(f'results_small_new/276/0Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', curflow_276_0)
    np.save(f'results_small_new/276/1Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', curflow_276_1)
    np.save(f'results_small_new/276/2Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', curflow_276_2)
    np.save(f'results_small_new/276/3Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', curflow_276_3)
    np.save(f'results_small_new/276/4Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', curflow_276_4)
    np.save(f'results_small_new/276/0Main{mainlane_demand}_merge{merge_lane_demand}insspeed2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', v_276_0s)
    np.save(f'results_small_new/276/1Main{mainlane_demand}_merge{merge_lane_demand}insspeed2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', v_276_1s)
    np.save(f'results_small_new/276/2Main{mainlane_demand}_merge{merge_lane_demand}insspeed2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', v_276_2s)
    np.save(f'results_small_new/276/3Main{mainlane_demand}_merge{merge_lane_demand}insspeed2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', v_276_3s)
    np.save(f'results_small_new/276/4Main{mainlane_demand}_merge{merge_lane_demand}insspeed2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', v_276_4s)





    np.save(f'results_small_new/Main{mainlane_demand}_merge{merge_lane_demand}average_density2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', densities)
    np.save(f'results_small_new/Main{mainlane_demand}_merge{merge_lane_demand}speeds2730_1.2_{controller}_lcCoop{args.p}_sim1_freedepart.npy', avg_speeds)

print('total travel time (h):',total_travel_time)
print('average bottleneck speed:',np.mean(avg_speeds))
print('average emission: ','CO:',np.mean(cos),'HC:',np.mean(hcs),'NOX:',np.mean(noxs),'PMX:',np.mean(pmxs))
env.close()