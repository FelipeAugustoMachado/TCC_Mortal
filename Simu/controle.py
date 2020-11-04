# -*- coding: utf-8 -*-
"""Controle_TCC-AM"""

#----- Importing Libraries -----#

import numpy as np
import matplotlib.pyplot as plt


#----- Class to Simulate Low Level Control -----#

class LLC:
  """
  Class to simulate the Low Level Control of the project.
  """

  def __init__(self, ts=0.05, tol=0.017):
    """
    Initializes all attributes used for the Low Level Controler calculations.
    The attributes are as follow:
      float y : system output
      float y_ant : previous system output
      float y_ant2 : 2 steps previous system output
      float u : control effort
      float u_ant : previous control effort
      float u_ant2 : 2 steps previous control effort
      float e : error
      float e_ant : previous error
      float e_ant2 : 2 steps previous error
      float r : controler reference
      float r_ant : previous controller reference
      float r_ant2 : 2 steps previous controller reference
      float r1 : pre-filter reference (system input reference)
      array float y_evolution : store temporal evolution of y
      array float u_evolution : store temporal evolution of u
      array float e_evolution : store temporal evolution of e
      array float r_evolution : store temporal evolution of r
      array float t_evolution : store measuring times of experiment
      bool target_reached : say if control target was reached
      float tol : tolerance in radians to determine when target is reached
      float Ts : sampling period in seconds
      float t : total experiment time so far

    Input:
      float ts : desired sampling period. Default is 50 ms.
      float tol : desired tolerance to stop control. Default is approximately 1
                  degree.
    
    Output:
      None
    """
    # Initialization of control variables:
    self.y = 0
    self.y_ant = 0
    self.y_ant2 = 0
    self.u = 0
    self.u_ant = 0
    self.u_ant2 = 0
    self.e = 0
    self.e_ant = 0
    self.e_ant2 = 0
    self.r = 0
    self.r_ant = 0
    self.r_ant2 = 0
    self.r1 = 0

    # Initialization for data storage:
    self.y_evolution = []
    self.u_evolution = []
    self.e_evolution = []
    self.r_evolution = []
    self.t_evolution = []

    # Initialization of other values:
    self.target_reached = False
    self.tol = tol
    self.Ts = ts
    self.t = 0


  def Do_LLC(self, ref=1.047):
    """
    Implements the Low Level Controler and simulates the effect on the plant.
    Receives an angular position reference (in radians) from the High Level
    Controler and stores the system output (angular position in radians),
    alongside with internal variables of interest for monitoring the simulation.

    Input:
      float ref : system input reference (angular position) in radians. Default
                  is 60 degrees.
    
    Output:
      None
    """
    self.r1 = ref #set desired system reference

    # Execute control until target is reached (considering tolerance):
    while not self.target_reached:
      #*** Pre-filter ***#

      #Calculating control reference:
      self.r = (25*self.r1*self.Ts**2 + self.r_ant*(2 + 20*self.Ts) - self.r_ant2*(1 + 5*self.Ts) )/(1 + 15*self.Ts + 25*self.Ts**2)
      
      #*** Controler ***#
      
      #Calculate error:
      self.e = self.r - self.y #sensor reading is system output
      #Calculating control effort (actuation output -> electric tension):
      self.u = (self.e*(73.602 + 2*self.Ts) - 98.136*self.e_ant + 24.534*self.e_ant2 + 4*self.u_ant - self.u_ant2)/(3 + 2*self.Ts)

      #*** Plant ***#

      #Calculating output:
      self.y = (12.588*self.u*self.Ts**2 + self.y_ant*(2 + 22.336*self.Ts) - self.y_ant2*(1 + 5.584*self.Ts))/(1 + 16.752*self.Ts)

      #*** Update variables ***#

      self.r_ant2 = self.r_ant
      self.r_ant = self.r
      self.e_ant2 = self.e_ant
      self.e_ant = self.e
      self.u_ant2 = self.u_ant
      self.u_ant = self.u
      self.y_ant2 = self.y_ant
      self.y_ant = self.y

      #*** Store data ***#

      self.y_evolution.append(self.y)
      self.u_evolution.append(self.u)
      self.e_evolution.append(self.e)
      self.r_evolution.append(self.r)
      self.t += self.Ts #update total experiment time
      self.t_evolution.append(self.t)

      #*** Check Target ***#

      if (self.y < self.r1 + self.tol) and (self.y > self.r1 - self.tol): #target reached
        self.target_reached = True


  def Reset_LLC(self):
    """
    Resets all instance variables back to the initialization values.
    """
    # Resetting control variables:
    self.y = 0
    self.y_ant = 0
    self.y_ant2 = 0
    self.u = 0
    self.u_ant = 0
    self.u_ant2 = 0
    self.e = 0
    self.e_ant = 0
    self.e_ant2 = 0
    self.r = 0
    self.r_ant = 0
    self.r_ant2 = 0
    self.r1 = 0

    # Resetting data storage:
    self.y_evolution = []
    self.u_evolution = []
    self.e_evolution = []
    self.r_evolution = []
    self.t_evolution = []

    # Resetting other values:
    self.target_reached = False
    self.t = 0  
    

  def ShowResults_LLC(self):
    """
    Print the results of the last experiment.
    """
    # Print system output y:
    plt.figure(1)
    plt.plot(self.t_evolution, self.y_evolution, 'b', label='y')
    plt.plot(self.t_evolution, self.r_evolution, 'm', label='r')
    plt.title('Evolution of Exoskeletons Angular Position')
    plt.ylabel('Angular Position [rad]')
    plt.xlabel('Time of Experiment [s]')
    plt.legend()
    plt.show()

    # Print systems control effort u:
    plt.figure(2)
    plt.plot(self.t_evolution, self.u_evolution, 'r', label='u')
    plt.title('Evolution of Control Effort')
    plt.ylabel('Electric Tension [V]')
    plt.xlabel('Time of Experiment [s]')
    plt.legend()
    plt.show()

    # Print system error e:
    plt.figure(3)
    plt.plot(self.t_evolution, self.e_evolution, 'k', label='e')
    plt.title('Evolution of Control Error')
    plt.ylabel('Error [rad]')
    plt.xlabel('Time of Experiment [s]')
    plt.legend()
    plt.show()

    # Print control reference r:
    plt.figure(4)
    plt.plot(self.t_evolution, self.r_evolution, 'm', label='r')
    plt.title('Evolution of Control Reference')
    plt.ylabel('Control Reference [rad]')
    plt.xlabel('Time of Experiment [s]')
    plt.legend()
    plt.show()


#----- Class for High Level Control -----#

class HLC:
  """
  Class for the High Level Control of the project.
  """

  # Defining State Machine values (constants):
  FLEXION = 1
  IDLE = 0
  EXTENSION = -1


  def __init__(self, ts=0.05, tol=0.034):
    """
    Initializes all attributes used for the High Level Control.
    The attributes are as follow:
      LLC llc : low level control object
      int state : indicates in which state the system currently is. 1 is Flexion,
                  -1 is extension and 0 is Idle.
      bool is_controlling : indicate if system is actuating the exoskeleton

    Input:
      float ts : desired LLC sampling period. Default is 50 ms.
      float tol : desired LLC tolerance to stop control. Default is
                  approximately 2 degrees.
    
    Output:
      None
    """
    # Initialization of attributes:
    self.llc = LLC(ts, tol)
    self.state = self.EXTENSION #starting condition
    self.is_controlling = False


  def Do_HLC(self, task):
    """
    Implements the State Machine logic of the High level Controller. Receives a
    signal of the desired action/state transition and executes the action
    according to the systems current State. For this, the LLC object is used,
    simulating the desired plant behavior.

    Input:
      int task : desired transition to another State. Can be 1 (Flexion), 0
                 (Idle) or -1 (Extension).
    
    Output:
      None
    """
    # System can only act if not in middle of an actuation:
    if (~self.is_controlling):

      if (self.state == self.EXTENSION and task == self.FLEXION): #Can actuate over exoskeleton
        # Set flag to ignore other commands:
        self.is_controlling = True

        # Set variables to the expected values in the current State:
        self.llc.Reset_LLC()
        # Execute Low Level Control simulation:
        self.llc.Do_LLC()
        # Show control result:
        #self.llc.ShowResults_LLC()

        # Reset State flag:
        self.is_controlling = False
        # Change systems State:
        self.state = self.FLEXION

      elif (self.state == self.FLEXION and task == self.EXTENSION): #Can return to start condition
        # Set flag to ignore other commands:
        self.is_controlling = True

        # Set variables to the expected values in the current State:
        self.llc.Reset_LLC()
        # Execute Low Level Control simulation:
        self.llc.Do_LLC(-1.047) #going back to start
        # Show control result:
        #self.llc.ShowResults_LLC()

        # Reset State flag:
        self.is_controlling = False
        # Change systems State:
        self.state = self.EXTENSION

      else: print("Can't do this") #Do nothing
    
    else: print("Already doing, sorry") #Do nothing


  def PrintMyState(self):
    """
    Prints the current state of the system.
    """

    if (self.state == self.EXTENSION):
      print("EXTENSION")
    elif (self.state == self.FLEXION):
      print("FLEXION")
    else:
      print("IDLE")
