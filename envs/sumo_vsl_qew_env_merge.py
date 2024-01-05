import gym
import traci
import sumolib
import numpy as np
from collections import deque
from agents.controller import IDMController,GippsController
import os
from utils.utils import *
from envs.merge_add_veh import qew_merge_add_veh

class sumo_qew_env_merge():
	def __init__(self,mainlane_demand,merge_lane_demand):
		self.gui = False
		self.lane_ids = []
		self.max_steps = 6000
		self.curr_step = 0
		self.collision = False
		self.done = False
		self.control_horizon=60
		self.outID_9728=['9728_0loop','9728_1loop','9728_2loop']
		self.outID_9832=['9832_0loop','9832_1loop','9832_2loop']
		self.outID_9575=['9575_0loop','9575_1loop','9575_2loop']
		self.outID_9813=['9813_0loop']
		self.inID=['9575_0loop','9575_1loop','9575_2loop','9813_0loop']
		self.outID=['9728_0loop','9728_1loop','9728_2loop']
		self.state_detector = ['9712_{}loop'.format(i) for i in range(12)]
		self.bottleneck_detector=['9712_0loop','9712_1loop','9712_2loop','9712_3loop']
		self.VSLlist=['9712_0','9712_1','9712_2','9712_3']
		self.mainlane_demand=mainlane_demand
		self.mergelane_demand=merge_lane_demand


	def start(self, gui=False, network_conf="networks/merge_qew_metering/sumoconfig.sumo.cfg", network_xml='networks/merge_qew_metering/qew_mississauga_rd.net.xml'):
		self.gui = gui
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		self.curr_step = 0
		self.collision = False
		self.done = False
		self.stats_path='results/stat1.xml'
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


	def get_step_state(self,):
		'''
		Define a state as a vector of vehicles marco information
		'''
		state_occu = []
		for detector in self.state_detector:
			occup = traci.inductionloop.getLastStepOccupancy(detector)
			if occup == -1:
				occup = 0
			state_occu.append(occup)
			
		return np.array(state_occu)
	
	def calc_outflow(self):
		state = []
		statef = []
		for detector in self.outID:
			veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
			state.append(veh_num)
		for detector in self.inID:
			veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
			statef.append(veh_num)
		return np.sum(np.array(state)) - np.sum(np.array(statef))
		
	def calc_bottlespeed(self):
		speed = []
		for detector in self.bottleneck_detector:
			dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
			if dspeed < 0:
				dspeed = 5                                              
                #The value of no-vehicle signal will affect the value of the reward
				speed.append(dspeed)
		return np.mean(np.array(speed))

    #####################  the CO, NOx, HC, PMx emission  #################### 
	def calc_emission(self):
		vidlist = traci.edge.getIDList()
		co = []
		hc = []
		nox = []
		pmx = []
		for vid in vidlist:
			co.append(traci.edge.getCOEmission(vid))
			hc.append(traci.edge.getHCEmission(vid))
			nox.append(traci.edge.getNOxEmission(vid))
			pmx.append(traci.edge.getPMxEmission(vid))
		return [np.sum(np.array(co))/1000,np.sum(np.array(hc))/1000,np.sum(np.array(nox))/1000,np.sum(np.array(pmx))/1000] #g

	def compute_reward(self):
		'''
			Reward function is made of three elements:
			 - speed 
			 - emission
			 - throughput
			 Taken from Wu et al.
		'''
		# Rewards Parameters
		emission=self.calc_emission()
		bspeed=self.calc_bottlespeed()
		oflow=self.calc_outflow()
		

		return oflow,bspeed,emission
		


	def step(self, v,step):

		done=False

		state_overall = 0
		reward = 0
		co = 0
		hc = 0
		nox = 0
		pmx = 0
		oflow = 0
		bspeed = 0
		set_vsl(v,self.VSLlist)
		traci.trafficlight.setPhase("J0", 0)
		for i in range(self.control_horizon):
			traci.simulationStep()
			if traci.trafficlight.getPhase("J0") == 0:
				if len(list(traci.vehicle.getIDList()))>2: ## this is a logic to control traffic light
					traci.trafficlight.setPhase("J0", 1)
			qew_merge_add_veh(step*self.control_horizon+i,self.mainlane_demand,self.mergelane_demand)
			state_overall = state_overall + self.get_step_state()
			oflow_,bspeed_,emission=self.compute_reward()
			oflow = oflow + oflow_
			bspeed = bspeed + bspeed_
             # the reward is defined as the outflow 
			co = co + emission[0]# g
			hc = hc + emission[1]# g
			nox = nox + emission[2]# g
			pmx = pmx + emission[3]# g
			
		reward = reward + oflow/80 * 0.1 + bspeed/(30*self.control_horizon)*0.9
		
		return state_overall/self.control_horizon/100, reward,done, oflow,bspeed/self.control_horizon,[co,hc,nox,pmx]


	def reset(self, gui=False):

		self.start(gui)
		state_overall=self.get_step_state()
		return state_overall

	def close(self):
		traci.close(False)
