import sys
sys.path.append(r"E:\GUC\semester 8\codes\gym_pathfinding_master")
import gym
import gym_pathfinding
import math
import numpy as np
import matplotlib.pyplot as plt

class Robot(object) :
    """ Defines basic mobile robot properties """ 
    def __init__(self) :
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.theta = 0.0
        self.plot = False
        self._delta = 0.1 #sample time
        self.env_width = 4
        self.env_length = 4

    # Movement
    def step (self):
        """ updates the x , y and angle """
        self.deltax()
        self.deltay()
        self.deltaTheta()

    def move(self , seconds):
        """ Moves the robot for an ' s ' amount of seconds """ 
        for i in range(int(seconds/self._delta)):
            self.step()
            if i%3 == 0 and self.plot: # plot path every 3 steps
                self.plot_xya ( )

    # Printing−and−plotting :
    def print_xya (self):
        """ prints the x , y position and angle """ 
        print("x = " + str(self.pos_x)+" "+"y = "+ str(self.pos_y))
        print("a = " + str(self.theta))

    def plot_robot (self):
        """ plots a representation of the robot """ 
        plt.arrow(self.pos_y ,self.pos_x , 0.001 * math.sin (self.theta ),0.001 * math.cos (self.theta ) ,
        head_width=self.length , head_length=self.length ,
        fc= 'k' , ec= 'k' )

    def plot_xya (self):
        """ plots a dot in the position of the robot """ 
        plt.scatter(self.pos_y ,self.pos_x,c= 'r',edgecolors= 'r' )

class MobileRobot (Robot ) :
    """ Defines a MobileRobot """ 
    def __init__(self):
        Robot.__init__(self)
        # geometric parameters
        self.r = 0.03
        self.c = 0.065
        self.a = 0.055
        self.b = 0.055
        self.Xicr = 0.055
        self.length = 0.11

        # states
        self.omega = 0
        self.vx = 0
        self.vy = 0

        self.omega_r = 0.0
        self.omega_l = 0.0

        self.orientation = 0
        
    def deltax(self) :
        # calculate vx and vy
        self.omega = self.r * (self.omega_r - self.omega_l)/(2* self. c )
        self.vx = self.r * (self.omega_r + self.omega_l)/2
        self.vy = self.Xicr * self.omega
        # calculate X_dot
        X_dot = math.cos(self.theta)*self.vx - math.sin(self.theta)*self.vy
        self. pos_x += self._delta * X_dot

        if self.pos_x > self.env_width:
            self.pos_x = self.env_width
        elif self.pos_x < 0:
            self.pos_x = 0

    def deltay ( self) :
        # calculate vx and vy
        self.omega = self.r * (self.omega_r - self.omega_l)/(2 * self.c)
        self.vx = self.r * (self.omega_r + self.omega_l)/2
        self.vy = self.Xicr * self.omega
        # calculate Y_dot
        Y_dot = math.sin ( self. theta ) * self.vx + math.cos ( self. theta ) * self. vy
        self. pos_y += self. _delta * Y_dot

        if self. pos_y > self.env_length:
            self. pos_y = self.env_length
        elif self. pos_y < 0:
            self. pos_y = 0

    def deltaTheta ( self) :
        # calculate omega
        self.omega = self. r * ( self. omega_r-self. omega_l ) /(2* self. c )
        self. theta += self. _delta * self.omega

    def reset (self,player) :
        # given 4mx4m arena discretized to 25x25
        self.pos_x = player[0] *self.env_width/24 #discrete levels
        self.pos_y = player[1] *self.env_length/24 #discrete levels
        self.theta = np.random.uniform(-np.pi,np.pi)
        # print(self.theta)

    def optimal_action ( self , action ) :
        self.action = action

    def take_step (self) :
        if self.action == 0 : # Forward
            self.omega_l = 1.7 
            self.omega_r = 1.7 
        elif self.action == 1 : # Backward
            self.omega_l = -1.7
            self.omega_r = -1.7
        elif self.action == 2 : # left
            self.omega_l = -1.7
            self.omega_r = 1.7 
        else : # right
            self.omega_l = 1.7 
            self.omega_r = -1.7
        self.move(self._delta)

    def get_discretized_state ( self) :
        x_discrete = math.floor ( ( ( self. pos_x ) /self.env_width) *24)
        y_discrete = math.floor ( ( ( self. pos_y ) /self.env_length) *24)
        theta_discrete = np.arctan2 (math. sin ( self. theta ) , math. cos ( self. theta ) )
        # print(theta_discrete *180/np.pi)
        theta_discrete = math.floor(theta_discrete/(2 * np.pi) * 20)
        # print(theta_discrete)

        return ( x_discrete , y_discrete , theta_discrete )

    def assign_discretized_state ( self , x , y ) :
        self.pos_x= ( ( x ) /24) *self.env_width
        self.pos_y= ( ( y ) /24) *self.env_length