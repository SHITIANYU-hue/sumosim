import gym
import traci
import sumolib
import numpy as np
from collections import deque
from agents.controller import IDMController,GippsController,secrmController
import os
from utils.utils import *




class sumo_env_merge():
	def __init__(self):
		self.name = 'rlagent'
		self.step_length = 0.4
		self.acc_history = deque([0, 0], maxlen=2)
		self.grid_state_dim = 4 ## why it has to be 4??
		self.state_dim = (4*self.grid_state_dim*self.grid_state_dim)+1 # 5 info for the agent, 4 for everybody else
		self.pos = (0, 0)
		self.curr_lane = ''
		self.curr_sublane = -1
		self.target_speed = 0
		self.speed = 0
		self.lat_speed = 0
		self.acc = 0
		self.angle = 0
		self.gui = False
		self.lane_ids = []
		self.rl_names = []
		self.rl_vehicles_state = {}
		self.rl_vehicles_grid_state={}
		self.rewards={}
		self.rl_vehicles_action={}
		self.max_steps = 6000
		self.curr_step = 0
		self.collision = False
		self.done = False


	def start(self, gui=False, warm_up_step=1000,network_conf="networks/merge_synth_small/sumoconfig.sumo.cfg", network_xml='networks/merge_synth_small/merge.net.xml'):
		self.gui = gui
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		self.curr_step = 0
		self.warm_up_step=warm_up_step
		self.collision = False
		self.done = False
		self.lane_change_model = 0b00100000000 ## disable lane change
		self.speed_mode=32 ## disable all check for speed
		self.stats_path='results_small/stat1.xml'
		# Starting sumo
		home = os.getenv("HOME")

		if self.gui:
			sumoBinary = home + "/code/sumo/bin/sumo-gui"
		else:
			sumoBinary = home + "/code/sumo/bin/sumo"
		sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
		save_stats=True
		if save_stats:
			sumoCmd.extend(["--statistic-output", self.stats_path, "--duration-log.statistics"])
		traci.start(sumoCmd)

		self.lane_ids = traci.lane.getIDList()

		# for step in range(self.warm_up_step):
		# 	traci.simulationStep()
		# Setting the lane change mode to 0 meaning that we disable any autonomous lane change and collision avoidance
		for i in range(len(self.rl_names)):
		# Setting up useful parameters
			self.update_params(self.rl_names[i])

	def update_params(self,name):
		# initialize params
		self.pos = traci.vehicle.getPosition(name)
		self.curr_lane = traci.vehicle.getLaneID(name)


		self.curr_sublane = int(self.curr_lane.split("_")[1])

		self.target_speed = traci.vehicle.getAllowedSpeed(name)
		self.speed = traci.vehicle.getSpeed(name)
		self.lat_speed = traci.vehicle.getLateralSpeed(name)
		self.acc = traci.vehicle.getAcceleration(name)
		self.acc_history.append(self.acc)
		self.angle = traci.vehicle.getAngle(name)


	# Get grid like state
	def get_grid_state(self, name, threshold_distance=20):
		'''
		Observation is a grid occupancy grid
		'''
		agent_lane = self.curr_lane
		agent_pos = self.pos
		edge = self.curr_lane.split("_")[0]
		agent_lane_index = self.curr_sublane
		lanes = [lane for lane in self.lane_ids if edge in lane]
		state = np.zeros([self.grid_state_dim, self.grid_state_dim], dtype=object)
		# Putting agent
		agent_x, agent_y = 1, agent_lane_index
		state[agent_x, agent_y] = -1
		# Put other vehicles
		for lane in lanes:
			# Get vehicles in the lane
			vehicles = traci.lane.getLastStepVehicleIDs(lane)
			veh_lane = int(lane.split("_")[-1])
			for vehicle in vehicles:
				if vehicle == name:
					continue
				# Get angle wrt rlagent
				veh_pos = traci.vehicle.getPosition(vehicle)
				# If too far, continue
				if get_distance(agent_pos, veh_pos) > threshold_distance:
					continue
				rl_angle = traci.vehicle.getAngle(name)
				# veh_id = vehicle.split("_")[1]
				veh_id = vehicle
				angle = angle_between(agent_pos, veh_pos, rl_angle)
				# Putting on the right
				if angle > 337.5 or angle < 22.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the right north
				if angle >= 22.5 and angle < 67.5:
					state[agent_x-1,veh_lane] = veh_id
				# Putting on north
				if angle >= 67.5 and angle < 112.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left north
				if angle >= 112.5 and angle < 157.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left
				if angle >= 157.5 and angle < 202.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the left south
				if angle >= 202.5 and angle < 237.5:
					state[agent_x+1, veh_lane] = veh_id
				if angle >= 237.5 and angle < 292.5:
					# Putting on the south
					state[agent_x+1, veh_lane] = veh_id
				# Putting on the right south
				if angle >= 292.5 and angle < 337.5:
					state[agent_x+1, veh_lane] = veh_id
		# Since the 0 lane is the right most one, flip 
		state = np.fliplr(state)
		return state
		
	def compute_jerk(self):
		return (self.acc_history[1] - self.acc_history[0])/self.step_length

	def detect_collision(self,name):
		collisions = traci.simulation.getCollidingVehiclesIDList()
		if name in collisions:
			self.collision = True
			return True
		self.collision = False
		return False
	
	def get_state(self,name):
		'''
		Define a state as a vector of vehicles information
		'''
		state = np.zeros(self.state_dim)
		before = 0
		grid_state = self.get_grid_state(name).flatten()
		for num, vehicle in enumerate(grid_state):
			# if vehicle  >=9:
			# 	vehicle=0 ## not sure why will greater than 9? (i think it is unnecessary)
			if vehicle == 0:
				continue
			if vehicle == -1:
				vehicle_name = self.name
				before = 1
			else:
				vehicle_name = vehicle ## this shall be changed
			veh_info = self.get_vehicle_info(vehicle_name)
			idx_init = num*4
			if before and vehicle != -1:
				idx_init += 1
			idx_fin = idx_init + veh_info.shape[0]
			state[idx_init:idx_fin] = veh_info
		state = np.squeeze(state)
		return state
	
	
	def get_vehicle_info(self, vehicle_name):
		'''
			Method to populate the vector information of a vehicle
		'''
		if vehicle_name == self.name:
			return np.array([self.pos[0], self.pos[1], self.speed, self.lat_speed, self.acc])
		else:
			lat_pos, long_pos = traci.vehicle.getPosition(vehicle_name)
			long_speed = traci.vehicle.getSpeed(vehicle_name)
			acc = traci.vehicle.getAcceleration(vehicle_name)
			dist = get_distance(self.pos, (lat_pos, long_pos))
			lane=traci.vehicle.getLaneIndex(vehicle_name)
			return np.array([dist, long_speed, acc, lat_pos])
		
		
	def compute_reward(self, collision, action,name,reward_type='secrm'):
		'''
			Reward function is made of three elements:
			 - Comfort 
			 - Efficiency
			 - Safety
			 Taken from Ye et al.
		'''
		# Rewards Parameters
		if reward_type=='ye':
			alpha_comf = 0.1
			w_speed = 5
			w_change = 1
			w_eff = 1
			
			# Comfort reward 
			jerk = self.compute_jerk()
			R_comf = -alpha_comf*jerk**2
			action[0]=map_action(action[0])
			#Efficiency reward
			# Speed
			R_speed = -np.abs(self.speed - self.target_speed)
			# Penalty for changing lane
			if action[0]!=0:
				R_change = -1
			else:
				R_change = 0
			# Eff
			# R_eff = w_eff*(w_speed*R_speed + w_change*R_change) ## i didn't add R_lane rihgt not since it is not mandatory lane change
			R_eff = R_speed
			# Safety Reward
			# Just penalize collision for now
			if collision:
				R_safe = -10
			else:
				R_safe = +1
			
			# total reward
			R_tot = R_comf + R_eff + R_safe

		if reward_type=='secrm':
			alpha_comf = 1
			w_speed = 1
			w_change = 0
			w_eff = 1
			w_safe=0
			ae=3
			de=5
			tau=0.1
			gamma=0.99

			this_vel,lead_vel,lead_info,headway,target_speed=self.get_ego_veh_info(name)
			if this_vel<-5000:
				this_vel=0

			this_vel=traci.vehicle.getSpeed(name)

			info=[this_vel,target_speed,headway,lead_vel]
			# print('target speed',target_speed)
			secrm_speed=secrmController().get_speed(info)
			secrm_2dspeed=secrmController().get_2d_speed(info,name)
			if max(secrm_2dspeed)>secrm_speed:
				max_speed=max(secrm_2dspeed)
				T=int((abs(max_speed-this_vel)/ae)*(1/tau))
				cost=(1-gamma**T)/(1-gamma)
				R_disc=cost*(this_vel-max_speed)/max_speed
			else:
				R_disc=0

			distance2=abs(traci.vehicle.getDrivingDistance(name,'276',0))
			sublane=self.curr_lane.split("_")[-1]
			if distance2<300:
				exit_dis=distance2
			else:
				exit_dis=0
			R_exit=0
			if sublane=='0':
				R_exit=-1/(1+0.01*exit_dis)
			# target_speed= controller.get_speed(info)
			# print('gipps speed',gipps_speed)
			# or R_speed = -np.abs(this_vel - target_speed)
   


			R_speed = -np.abs(this_vel - secrm_speed)/secrm_speed
			# print('R speed',R_speed)
			if action[0]!=0:
				R_change = -1
			else:
				R_change = 0
			# Eff
			R_eff = w_eff*(w_speed*R_speed + w_change*R_change) ## i didn't add R_lane rihgt now since it is not mandatory lane change

			if collision:
				R_safe = -10
			else:
				R_safe = +1

			R_safe=w_safe*R_safe
			jerk = self.compute_jerk()
			R_comf = -alpha_comf*(jerk/(ae+de))**2
			# print('r safe',R_safe,'r eff',R_eff,'r comfort', R_comf)
			R_tot = R_comf + R_eff + R_safe
			# print('R total',R_tot)
			# print('Rcomf',R_comf,'Reff',R_eff,'Rdisc',R_disc,'R_exit',R_exit)

		return [R_tot, R_comf, R_eff, R_disc,R_exit]
		

	def apply_acceleration(self, vid, acc, smooth=True):
		"""See parent class."""
		# to handle the case of a single vehicle
		
		this_vel = traci.vehicle.getSpeed(vid)
		next_vel = max([this_vel + acc * 0.1, 0])
		if smooth:
			traci.vehicle.slowDown(vid, next_vel, 1e-3)
		else:
			traci.vehicle.setSpeed(vid, next_vel)

	def get_ego_veh_info(self,name):
		
		lead_info = traci.vehicle.getLeader(name)
		trail_info = traci.vehicle.getFollower(name)
		this_vel=traci.vehicle.getSpeed(name)
		target_speed=traci.vehicle.getAllowedSpeed(name)
		target_speed=30

		if lead_info is None or lead_info == '' or lead_info[1]>5:  # no car ahead??
			s_star=0
			headway=999999
			lead_vel=99999
		else:
			lead_id=traci.vehicle.getLeader(name)[0]
			headway = traci.vehicle.getLeader(name)[1]
			lead_vel=traci.vehicle.getSpeed(lead_id)

		return this_vel,lead_vel,lead_info,headway,target_speed


	def calculate_distance_veh(self,lead_info,ego_info):
		veh_pos = traci.vehicle.getPosition(lead_info)
		ego_pos = traci.vehicle.getPosition(ego_info)
		headway=get_distance(veh_pos,ego_pos)
		return headway


	def get_rela_ego_veh_info(self,name,veh_id):
		
		target_speed=traci.vehicle.getAllowedSpeed(name)

		if veh_id ==0:  # no car ahead
			headway=999999
			lead_vel=target_speed
		else:
			try:
				lead_vel=traci.vehicle.getSpeed(veh_id) ##sometimes sumo can't find leader veh if it is too far?
			except:
				lead_vel=target_speed

		return [target_speed,lead_vel]

	def compute_action(self,action,name,max_dec=-3,max_acc=3,stop_and_go=False,sumo_lc=False,sumo_carfollow=False,lane_change='SECRM',car_follow='Gipps'):
		this_vel,lead_vel,lead_info,headway,target_speed=self.get_ego_veh_info(name)
		# print('this vel',this_vel)
		# print('action',action,'name',name)
		# print('headway',headway)

		# action[0]=map_action(action[0])

		# action[1]=max(max_dec, min(action[1], max_acc))

		parts=self.curr_lane.split("_")
		if len(parts)>=2:
			edge="_".join(parts[0:2])
		if len(parts)==2:
			edge=parts[0]
		num_lanes = traci.edge.getLaneNumber(edge)

		if action[0] != 0:
			if action[0] == 1 and parts[0]!='276' and parts[1]!='1': ## change to right (not allow to change right to merge lane)

				# if self.curr_sublane >=1 :
				# 	try:
				# 		traci.vehicle.changeLane(name, self.curr_sublane-1, 0.1)
				# 	except:
				# 		pass
				pass
				# traci.vehicle.changeLaneRelative(name, -1,1)

			if action[0] == 2: ## change to left
				# if self.curr_sublane < num_lanes-1:
				# 	try:
				# 		traci.vehicle.changeLane(name, self.curr_sublane + 1, 0.1)
				# 	except:
				# 		pass
				traci.vehicle.changeLaneRelative(name, 1, 1)

		self.apply_acceleration(name,action[1])

			



	def step(self, action,veh_id_list,max_dec=-3,max_acc=3,stop_and_go=False,sumo_lc=False,sumo_carfollow=False,lane_change='SECRM',car_follow='Gipps'):
		'''
		This will :
		- send action, namely change lane or stay 
		- do a simulation step
		- compute reward
		- update agent params 
		- compute nextstate
		- return nextstate, reward and done
		'''
		done=False
		for i in range(len(veh_id_list)):
			self.compute_action(action[veh_id_list[i]],veh_id_list[i],max_dec=max_dec,max_acc=max_acc,stop_and_go=stop_and_go,sumo_lc=sumo_lc,sumo_carfollow=sumo_carfollow,lane_change=lane_change,car_follow=car_follow)


		collision=False
		# action[0]=map_to_minus_zero_plus(action[0])
		# Check collision
		for i in range(len(veh_id_list)):
			collision_i = self.detect_collision(veh_id_list[i])
			if collision_i:
				print('collision',collision_i,'name',veh_id_list[i])
				collision=True
				break
			else:
				collision=False
		# Compute Reward
		reward=0
		for i in range(len(veh_id_list)):
			reward_i = self.compute_reward(collision, action[veh_id_list[i]],veh_id_list[i])
			reward+=reward_i[0]
			self.rewards[veh_id_list[i]]=reward_i

		# Sim step
		traci.simulationStep()

		veh_id_list=list(traci.vehicle.getIDList()) ## when it compute the action some vehicle will run out 

		# Update agent params 
		if not collision and not done:
			for i in range(len(veh_id_list)):
				self.update_params(veh_id_list[i])
		# State
		next_state=[]
		for i in range(len(veh_id_list)):
			next_state = self.get_state(veh_id_list[i]) ##length is 65
			self.rl_vehicles_grid_state[veh_id_list[i]]=self.get_grid_state(veh_id_list[i])
			self.rl_vehicles_state[veh_id_list[i]]=next_state
		# Update curr state
		self.curr_step += 1
		next_state=self.rl_vehicles_state
		next_grid_state=self.rl_vehicles_grid_state
		if self.curr_step <= self.max_steps:
			done = collision
		else:
			done = True
			self.curr_step = 0

		return next_grid_state,next_state, self.rewards, done, collision
		
	def render(self, mode='human', close=False):
		pass

	def reset(self, gui=False):

		self.start(gui)
		veh_id_list=list(traci.vehicle.getIDList())
		for i in range(len(veh_id_list)):
			self.rl_vehicles_state[veh_id_list[i]]=self.get_state(veh_id_list[i])
		# print('type',self.rl_vehicles_state)
		return np.zeros(65)


	def close(self):
		traci.close(False)
