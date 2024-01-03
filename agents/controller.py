import numpy as np



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
                 tau=0.1,#把这个参数改到了1之后更容易发生碰撞了
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
        target_speed,lead_vel,headway,this_vel=info

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
        target_speed,lead_vel,this_vel,v_next =info
        # v_next=self.get_speed(info)
        acceleration= (v_next-this_vel)/self.sim_step

        return acceleration