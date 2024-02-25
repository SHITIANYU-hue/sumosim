from configparser import ConfigParser
from argparse import ArgumentParser
import traci
import numpy as np
import os
import random
from envs.synthetic_small_env_merge import sumo_env_merge
from envs.add_veh import syn_merge_add_veh
import json
from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG
import time 
from utils.utils import *
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default ='gym_sumo_qew_env_merge')
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--manual', type=bool, default=False, help="(default: False)")
parser.add_argument('--horizon', type=int, default=18000*0.5, help='number of simulation steps, (default: 6000)')
parser.add_argument('--coop', type=float, default=0, help='cooperative factor for human vehicles')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 10, help = 'save interval(default: 100)')
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

action_dim = 2
state_dim = 65
state_rms = RunningMeanStd(state_dim)
exp_tag='sacsmallmerge'
unix_timestamp = int(time.time())


    
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

    
if (torch.cuda.is_available()) and (args.use_cuda):
    agent = agent.cuda()

if args.load != 'no':
    agent.load_state_dict(torch.load("./model_weights/"+args.load))



    
env = sumo_env_merge()



outID_250=['250_0loop','250_1loop']
outID_246=['246_0loop','246_1loop']
outID_319=['319_0loop','319_1loop']
outID_994=['994_0loop']



# for i in range(len(flow_rates)):




mainlane_demand = 0.5*3600 ##5500, 4400,3850,3300
merge_lane_demand = 0
interval = 2000*0.1  # interval for calculating average statistics
simdur = args.horizon  # assuming args.horizon represents the total simulation duration
curflow = 0
curflow_994 = 0
curflow_246 = 0
curflow_319 = 0
score_lst = []
avg_scors=[]

state_lst=[]
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
warmup=100

if agent_args.on_policy == True:
    score = 0.0
    score_comfort=0
    score_eff=0
    score_safe=0
    action={}
        # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        states = env.reset(gui=args.render)
        t=1
        print('epoch',n_epi)
        while t < simdur:
            print('step', t)
            try:
                veh_id_list=syn_merge_add_veh(t,mainlane_demand,merge_lane_demand) ## it could be possible multiple vehicles are reapeated added
            except:
                pass
            print('veh id list!',veh_id_list)
            vehPermid = get_vehicle_number('276_0') + get_vehicle_number('276_1') + get_vehicle_number('276_2')
            vehPerout = get_vehicle_number('250_0') + get_vehicle_number('250_1') 

            # veh_number_total=traci.vehicle.getIDCount()+int(len(traci.simulation.getPendingVehicles())) ##i find that it will make it run very slow
            # veh_number_total=traci.vehicle.getIDCount()+len(traci.edge.getPendingVehicles('994')) ## this is to count waiting vehicle in the merge
            veh_number_total=traci.vehicle.getIDCount()

            total_travel_time = total_travel_time+ (time_step*veh_number_total)/3600

            avg_speed=(get_meanspeed('276_0')+get_meanspeed('276_1')+get_meanspeed('276_2'))/3 ### this is only bottlneck's speed
            print('avg speed',avg_speed)

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
            if t>warmup:
                actions={}
                for veh_id in veh_id_list:
                    state=states[veh_id]
                    state_lst.append(state)
                    mu,sigma = agent.get_action(torch.from_numpy(state).float().to(device))
                    dist = torch.distributions.Normal(mu,sigma[0])
                    action = dist.sample()
                    action_=action.cpu().detach().numpy()
                    actions[veh_id]=action_
                    log_prob = dist.log_prob(action).sum(-1,keepdim = True)
                next_states, reward_info, done, info = env.step(actions,veh_id_list)
                print('reward!',reward_info)

                for veh_id in veh_id_list:
                    Reward=reward_info[veh_id]
                    if len(reward_info)==0:
                        reward, R_comf, R_eff, R_safe=0,0,0,0
                    else:
                        reward, R_comf, R_eff, R_safe = Reward
                    state=states[veh_id]
                    next_state=next_states[veh_id]
                    transition = make_transition(state,\
                                                action.cpu().numpy(),\
                                                np.array([reward*args.reward_scaling]),\
                                                next_state,\
                                                np.array([done]),\
                                                log_prob.detach().cpu().numpy()\
                                                )
                    agent.put_data(transition) 
                    score += reward
                    score_comfort +=R_comf
                    score_eff += R_eff
                    score_safe +=R_safe
                if done:
                        # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                        score_lst.append(score)
                        if args.tensorboard:
                            writer.add_scalar("score/score", score, n_epi)
                            writer.add_scalar("score/comfort", score_comfort, n_epi)
                            writer.add_scalar("score/safe", score_safe, n_epi)
                            writer.add_scalar("score/eff", score_eff, n_epi)

                        score = 0
                        score_comfort=0
                        score_eff=0
                        score_safe=0
                        # env.close()
                        break

                else:
                    states = next_states
                    # state_ = next_state_
        score_lst.append(score)
        agent.train_net(n_epi)
        print('state list',state_lst)
        state_rms.update(np.vstack(state_lst))
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst))) ## sometimes it will divide by zero
            avg_scors.append(sum(score_lst)/len(score_lst))
            print('avg scores',avg_scors)
            np.save(f'score/avgscores_{exp_tag}_{args.algo}_{unix_timestamp}.npy',avg_scors)
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),f'./model_weights/agent_{exp_tag}_{args.algo}'+str(n_epi))

            
            # # # # Save the average values
            # np.save(f'results_small/250/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows)
            # np.save(f'results_small/994/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_994)
            # np.save(f'results_small/246/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_246)
            # np.save(f'results_small/319/Main{mainlane_demand}_merge{merge_lane_demand}average_flow2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', inflows_319)
            # np.save(f'results_small/Main{mainlane_demand}_merge{merge_lane_demand}average_density2730_1.2_idm_lcCoop{args.coop}_lcStrategic1_freedepart.npy', densities)
        print('total travel time (h):',total_travel_time)
        print('average bottleneck speed:',np.mean(avg_speeds))
        print('average emission: ','CO:',np.mean(cos),'HC:',np.mean(hcs),'NOX:',np.mean(noxs),'PMX:',np.mean(pmxs))
        env.close()



else : # off policy 
    score = 0.0
    score_comfort=0
    score_eff=0
    score_safe=0
    pre_step=20
    for n_epi in range(args.epochs):
        state = env.reset(gui=False, numVehicles=25)
        done = False

        t=1
        while t < simdur:
            lane = 0
            acceleration = 0  # 0 for no change, 1 for acceleration, -1 for deceleration
            veh_id_list=syn_merge_add_veh(t,mainlane_demand,merge_lane_demand)
            vehPermid = get_vehicle_number('276_0') + get_vehicle_number('276_1') + get_vehicle_number('276_2')
            vehPerout = get_vehicle_number('250_0') + get_vehicle_number('250_1') 
            # veh_number_total=traci.vehicle.getIDCount()+int(len(traci.simulation.getPendingVehicles())) ##i find that it will make it run very slow
            # veh_number_total=traci.vehicle.getIDCount()+len(traci.edge.getPendingVehicles('994')) ## this is to count waiting vehicle in the merge
            veh_number_total=traci.vehicle.getIDCount()

            total_travel_time = total_travel_time+ (time_step*veh_number_total)/3600

            avg_speed=(get_meanspeed('276_0')+get_meanspeed('276_1')+get_meanspeed('276_2'))/3 ### this is only bottlneck's speed
            print('avg speed',avg_speed)

            if args.render:    
                env.render()
            action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
            if args.algo=='sac': ##not sure why action generate by sac needs to take out
                action=action[0]
            action = action.cpu().detach().numpy()
            next_state_, reward_info, done, collision = env.step(action, sumo_lc=True, sumo_carfollow=False, stop_and_go=False, car_follow='Gipps', lane_change='SECRM')
            
            if done or collision:
                # state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if args.tensorboard:
                    writer.add_scalar("score/score", score, n_epi)
                    writer.add_scalar("score/comfort", score_comfort, n_epi)
                    writer.add_scalar("score/safe", score_safe, n_epi)
                    writer.add_scalar("score/eff", score_eff, n_epi)
                score = 0
                score_comfort=0
                score_eff=0
                score_safe=0
                env.close()
                break              

            else:
                next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                state = next_state




            print('reward',reward_info)
            print('collision',collision)
            print('done',done)
            reward, R_comf, R_eff, R_safe,_ = reward_info
            transition = make_transition(state,\
                                         action,\
                                         np.array([reward*args.reward_scaling]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 
            score += reward
            score_comfort +=R_comf
            score_eff += R_eff
            score_safe +=R_safe


            if agent.data.data_idx > agent_args.learn_start_size: 
                agent.train_net(agent_args.batch_size, n_epi)


        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            avg_scors.append(sum(score_lst)/len(score_lst))
            print('avg scores',avg_scors)
            np.save(f'score/avgscores_{exp_tag}_{args.algo}_{unix_timestamp}.npy',avg_scors)
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),f'./model_weights/agent_{exp_tag}_{args.algo}'+str(n_epi))