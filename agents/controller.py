import numpy as np
import traci
from scipy.spatial import distance

def get_distance(a, b):
	return distance.euclidean(a, b)

def calculate_weighted_absolute_sum(list1, list2, lc_weight=2):
    """
    Calculate the sum of the absolute values of the elements of two lists,
    applying a customizable weight (default is 1.2) to the first element of each list.
    
    Parameters:
    - list1: First list of numbers.
    - list2: Second list of numbers.
    - lc_weight: Weight to apply to the first element of each list (default is 1.2).
    
    Returns:
    - float: The sum of all elements in both lists with the specified weight applied to the first elements,
             considering the absolute values of the elements.
    """
    # Apply the weight to the first element of each list and calculate the sum of absolute values
    weighted_sum = (lc_weight * abs(list1[0]) + sum(abs(x) for x in list1[1:])) + \
                   (lc_weight * abs(list2[0]) + sum(abs(x) for x in list2[1:]))
    
    return weighted_sum

def check_safe(Vego,VF_L,r=0.1,b_e=3,b_l_f=3,epsilon=0.1):
    gap=VF_L*r+(VF_L**2)/(2*b_l_f)-(Vego**2)/(2*b_e)
    # print('gap',gap)
    if gap> epsilon:
        return True
    else:
        return False

class IDMController:
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 T=0.1,
                 a=5,
                 b=1.5,
                 delta=2,
                 s0=2):
        """Instantiate an IDM controller."""
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

    def get_accel(self, info):
        """See parent class."""
        this_vel,target_speed,headway,lead_vel,lead_info=info
        if lead_info is None or lead_info == '':  # no car ahead
            s_star=0

        else:
            s_star = 2+ max(
                0, this_vel * self.T+ this_vel * (this_vel - lead_vel) /
                (2 * np.sqrt(self.a*self.b)))

        return 1 * (1 - (this_vel / target_speed)**self.delta - (s_star / headway)**self.s0)


class GippsController:
    """Gipps' Model controller.

    For more information on this controller, see:
    Traffic Flow Dynamics written by M.Treiber and A.Kesting
    By courtesy of Springer publisher, http://www.springer.com

    http://www.traffic-flow-dynamics.org/res/SampleChapter11.pdf

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    acc : float
        max acceleration, in m/s2 (default: 1.5)
    b : float
        comfortable deceleration, in m/s2 (default: -1)
    b_l : float
        comfortable deceleration for leading vehicle , in m/s2 (default: -1)
    s0 : float
        linear jam distance for saftey, in m (default: 2)
    tau : float
        reaction time in s (default: 1)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 v0=30,
                 acc=1.5, ## 增加这个参数一定程度可以优化jerk，同时会一定程度削弱eff，但是效果不是很明显
                 b=-1.5, ## 增加这个参数（如-1到-3）会导致其急停，进而导致削弱eff，此外会增加collision rate, 如果增加这个参数会使得自车更加靠近前车（但是容易相撞）
                 b_l=-0.8,
                 s0=4,
                 tau=1,#把这个参数改到了1之后更容易发生碰撞了
                 sim_step=0.1):
        """Instantiate a Gipps' controller."""

        self.v_desired = v0
        self.acc = acc
        self.b = b
        self.b_l = b_l
        self.s0 = s0
        self.tau = tau
        self.sim_step=sim_step

    def get_speed(self, info):
        """See parent class."""
        this_vel,target_speed,headway,lead_vel=info
        v_acc = this_vel + (2.5 * self.acc * self.tau * (
                1 - (this_vel / target_speed)) * np.sqrt(0.025 + (this_vel / target_speed)))

        v_safe = (self.tau * self.b) + np.sqrt(((self.tau**2) * (self.b**2)) - (
               self.b * ((2 * (headway-self.s0)) - (self.tau * this_vel) - ((lead_vel**2) /self.b_l))))

        # print('v acc',v_acc,'v safe',v_safe,'target',target_speed)
        v_next = min(v_acc, v_safe, target_speed)
        # v_next = min(v_acc, target_speed)

        return v_next
    def get_accel(self, info):
        """See parent class."""
        this_vel,target_speed,headway,lead_vel,lead_info=info

        v_acc = this_vel + (2.5 * self.acc * self.tau * (
                1 - (this_vel / target_speed)) * np.sqrt(0.025 + (this_vel / target_speed)))

        v_safe = (self.tau * self.b) + np.sqrt(((self.tau**2) * (self.b**2)) - (
               self.b * ((2 * (headway-self.s0)) - (self.tau * this_vel) - ((lead_vel**2) /self.b_l))))

        # print('v acc',v_acc,'v safe',v_safe,'target',target_speed)
        v_next = min(v_acc, v_safe, target_speed)
        acceleration=(v_next-this_vel)/self.sim_step

        return acceleration

class secrmController:
    def __init__(self,
                 v0=30,
                 acc=1.5, 
                 b=3,
                 a=3,
                 eps=4,
                 tau=0.1,#reaction time
                 sim_step=0.1):
        """Instantiate a secrm' controller."""

        self.v_desired = v0
        self.acc = acc
        self.b = b
        self.a = a
        self.eps = eps
        self.tau = tau
        self.sim_step=sim_step

    def get_speed(self, info):
        """See parent class."""
        this_vel,target_speed,headway,lead_vel=info
        gnew = headway - self.tau*(this_vel - lead_vel)
        vnew_sqrt = np.maximum((lead_vel + 0.5*self.b*self.tau)*(lead_vel + 0.5*self.b*self.tau) + 2*self.b*(headway - lead_vel*self.tau - self.eps) - self.b*self.tau*(this_vel - lead_vel), 0.0)
    #    vnew_sqrt = (w + 0.5*b*r)*(w + 0.5*b*r) + 2*b*(g - w*r - eps) - b*r*(v - w)
        vnew = -0.5 * self.b * self.tau + np.sqrt(vnew_sqrt)
        vnew = np.clip(vnew, 0.0, this_vel + self.tau*self.a)
        vnew = np.minimum(vnew, target_speed)
        # v_next = min(v_acc, target_speed)

        return vnew
    

    def get_accel(self, info):
        """See parent class."""
        this_vel,target_speed,headway,lead_vel=info
        gnew = headway - self.tau*(this_vel - lead_vel)
        vnew_sqrt = np.maximum((lead_vel + 0.5*self.b*self.tau)*(lead_vel + 0.5*self.b*self.tau) + 2*self.b*(headway - lead_vel*self.tau - self.eps) - self.b*self.tau*(this_vel - lead_vel), 0.0)
    #    vnew_sqrt = (w + 0.5*b*r)*(w + 0.5*b*r) + 2*b*(g - w*r - eps) - b*r*(v - w)
        vnew = -0.5 * self.b * self.tau + np.sqrt(vnew_sqrt)
        vnew = np.clip(vnew, 0.0, this_vel + self.tau*self.a)
        vnew = np.minimum(vnew, target_speed)
        # v_next = min(v_acc, target_speed)
        acceleration=(vnew-this_vel)/self.sim_step
        return acceleration



class coopsecrmController:
    def __init__(self,
                 v0=30,
                 acc=1.5, 
                 b=3,
                 a=3,
                 eps=4,
                 tau=0.1,#reaction time
                 sim_step=0.1,
                 p=0 ## p is the politeness factor
                 ): 
        """Instantiate a secrm' controller."""

        self.v_desired = v0
        self.acc = acc
        self.b = b
        self.a = a
        self.eps = eps
        self.tau = tau
        self.sim_step=sim_step
        self.p = p

    def get_speed(self, info):
        """See parent class."""
        this_vel,target_speed,headway,lead_vel=info
        gnew = headway - self.tau*(this_vel - lead_vel)
        vnew_sqrt = np.maximum((lead_vel + 0.5*self.b*self.tau)*(lead_vel + 0.5*self.b*self.tau) + 2*self.b*(headway - lead_vel*self.tau - self.eps) - self.b*self.tau*(this_vel - lead_vel), 0.0)
    #    vnew_sqrt = (w + 0.5*b*r)*(w + 0.5*b*r) + 2*b*(g - w*r - eps) - b*r*(v - w)
        vnew = -0.5 * self.b * self.tau + np.sqrt(vnew_sqrt)
        vnew = np.clip(vnew, 0.0, this_vel + self.tau*self.a)
        vnew = np.minimum(vnew, target_speed)
        # v_next = min(v_acc, target_speed)

        return vnew
    
    def calculate_distance_veh(self,lead_info,ego_info):
        veh_pos = traci.vehicle.getPosition(lead_info)
        ego_pos = traci.vehicle.getPosition(ego_info)
        headway=get_distance(veh_pos,ego_pos)
        return headway

    def get_delta_accel(self,name,new_follow):
        """
        input: vehicle name
        output other controller's potential accelration change
        """
        follow_info=traci.vehicle.getFollower(name)[0]
        action_new_follow=[0,0]
        # print('name',name,'new_follow',new_follow)
        if new_follow!='' and  new_follow!=follow_info:
            if type(new_follow)==tuple:
                action_ego_follow=[1,3]
                try:
                    action_new_follow=self.get_accel(new_follow[0][0])
                except:
                    action_ego_follow=[1,3] ## it will have collision!
            else:
                action_new_follow=self.get_accel(new_follow)

            
        if follow_info!='':
            action_ego_follow=self.get_accel(follow_info)
        else:
            action_ego_follow=[0,0]
        # print('name',name,'ego follow acc',action_ego_follow,'new follow acc',action_new_follow)

        ## we consider both change left and right have same impact on the environment
        if action_new_follow[0]!=0:
            action_new_follow[0]=1
        if action_ego_follow[0]!=0:
            action_ego_follow[0]=1

        return action_new_follow,action_ego_follow


    def get_accel(self,name):
        """
        input: vehicle name
        output controller's 2d action(rule based)
        """
        action=[0,0]
        # this_vel,target_speed,headway,lead_vel=info
        headway = traci.vehicle.getLeader(name)
        headway = 999 if headway is None else headway[1]

        target_speed=traci.vehicle.getAllowedSpeed(name)
        this_vel=traci.vehicle.getSpeed(name)
        lead_left= traci.vehicle.getNeighbors(name,'010')
        lead_right=traci.vehicle.getNeighbors(name,'011')
        follow_left=traci.vehicle.getNeighbors(name,'000')
        follow_right=traci.vehicle.getNeighbors(name,'001')
        lead_info=traci.vehicle.getLeader(name)
        follow_info=traci.vehicle.getFollower(name)

        if lead_left is None or len(lead_left) == 0:  # no car ahead
            lead_id=0
            lead_vel=target_speed
            headway_left=headway
        else:
            lead_id=lead_left[0][0]
            lead_vel=traci.vehicle.getSpeed(lead_id)
            headway_left=self.calculate_distance_veh(lead_id,name)
        info_n=[this_vel,target_speed,headway_left,lead_vel]
        speed_n= self.get_speed(info_n)
        if follow_left is None or len(follow_left)==0:
            safe_left=True
        else:
            safe_left=check_safe(this_vel,traci.vehicle.getSpeed(follow_left[0][0]))
        # safe_left=True
        if lead_right is None or len(lead_right) == 0 :  # no car ahead
            lead_id=0
            lead_vel=target_speed
            headway_right=headway

        else:
            lead_id=lead_right[0][0]
            lead_vel=traci.vehicle.getSpeed(lead_id)
            headway_right=self.calculate_distance_veh(lead_id,name)
        info_s=[this_vel,target_speed,headway_right,lead_vel]
        speed_s= self.get_speed(info_s)
        if follow_right is None or len(follow_right)==0:
            safe_right=True
        else:
            safe_right=check_safe(this_vel,traci.vehicle.getSpeed(follow_right[0][0]))

        if lead_info is None or lead_info == '' or lead_info[1]>30:  # no car ahead
            lead_id=0
            lead_vel=target_speed
            headway_e=headway
        else:
            lead_id=lead_info[0]
            lead_vel=traci.vehicle.getSpeed(lead_id)
            headway_e=self.calculate_distance_veh(lead_id,name)
        info_e=[this_vel,target_speed,headway_e,lead_vel]
        speed_e= self.get_speed(info_e)
        
        # print('headway',headway,'headwaye',headway_e,'headwayright',headway_right,'headwayleft',headway_left)

        change_right=traci.vehicle.couldChangeLane(name,-1) and safe_right
        change_left=traci.vehicle.couldChangeLane(name,1) and safe_left
        # print('safe left',safe_left,'safe right',safe_right,'changeleft',change_left)
        if speed_n>speed_e and speed_n >speed_s and change_left:
            action[0]=2 ## change left
        if speed_s>speed_e and speed_s > speed_n and change_right:
            action[0]=1 ## change right
        if abs(speed_n-speed_s)<100 and min(speed_n,speed_s)>speed_e:
            if change_right==True and change_left==False:
                action[0]=1
            if change_left ==True and change_right==False:
                action[0]=2
            if change_left ==True and change_right==True:
                action[0]=2

        if action[0]==2:
            vnew=speed_n
            new_follow=follow_left
        if action[0]==1:
            vnew=speed_s
            new_follow=follow_right
        else:
            vnew=speed_e
            new_follow=follow_info[0]
        a,b=self.get_delta_accel(name,new_follow)
        impact=calculate_weighted_absolute_sum(a,b)/5 ## 5 is because lc factor =2 , lc factor *1 + max acc(3)=5
        # v_next = min(v_acc, target_speed)

        action[1]=(vnew-this_vel)/self.sim_step

        prob=np.random.uniform(0, 1)
        if prob < self.p*impact: #(impact 0.2,0.3,0.5,1,2,5)
            action=[0,0]

        # print('action',action)
        return action

