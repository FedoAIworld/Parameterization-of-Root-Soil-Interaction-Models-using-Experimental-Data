"""
Copyright: 2021
Author: Fernand B. Eloundou
----------------------------
File: RootParamsOpt.py
"""


# Import libraries

import sys
sys.path.append("../../..")
sys.path.append("../../../src/python_modules")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vtk_plot as vp
import plantbox as pb
import pickle as pkl
import math
import numpy.matlib
from scipy.optimize import differential_evolution
from pyevtk.hl import gridToVTK                                 # pip3 install pyevtk
import time

# Path to and name of plant and root parameter file
path = "../../../modelparameter/rootsystem/"
name = "spring_barley"

"""
Optimize sensitive root parameters using Differential Evolution without subsoil treatment.
RMSE is the objective function with the measured root length density as the target variable.
The elongation rate is scaled using bulk density values based on the EquidistantGrid1D function in CPlantBox. 
"""

t = time.process_time()

simtime = 107              # simulation time in days
N = 8                      # number of plants in rows
M = 2                      # number of rows
distp = 3.                 # distance between the root systems along row[cm]
distr = 12.5               # distance between the rows[cm]
distTr = N*distr           # total distance between plants
distTp = M*distp           # total row spacing
top = 0.                   # vertical top position (cm) (normally = 0)
bot = -100.                # vertical bot position (cm) (e.g. = -100 cm)
left = -50.                # left along x-axis (cm)
right = 50.                # right along x-axis (cm)
n = 20                     # number of layers, each with a height of (top-bot)/n
m = 20                     # number of horizontal grid elements (each with length of (right-left)/m)
m2 = int(M * distp)  	   # resolution of 1 cm 
exact = True               # calculates the intersection with the layer boundaries (true), only based on segment midpoints (false)

# soil volume
soilVolume =   5*5*5

# Times correspond to days after sowing (DAS)
times = [74, 73, 71, 46, 45]

# Measured RLD
with open('RLD_ctrl.pkl','rb') as f:
        measured_RLD = pkl.load(f)


# Define bulk densities 
scale_elongation = pb.EquidistantGrid1D(0, -120, 9)  			  # 0=surface; -120=depth; 9=number of layer boundries
bulk_density = np.array([1.30, 1.50, 1.56, 1.50, 1.53, 1.57, 1.59, 1.61])

# Polynomial function
a=2.52244967; b=9.477859748; c=9.058277729; 
scales = a*bulk_density**2 - b*bulk_density + c

scale_elongation.data = scales


def err2(fitparams):
        """
        Creates 8*2 root systems and scale elongation rate.
        Calibrate sensitive root parameters using field RLD
        Returns the squared root of the mean square error between measured and simulated RLD
        """
        lmax0 		= fitparams[0]
        theta0 	        = fitparams[1]
        r0 		= fitparams[2]
        ln1 		= fitparams[3]
        tropismN0 	= fitparams[4]
        lb1 		= fitparams[5]
        la1 		= fitparams[6]
        lmax1 		= fitparams[7]
        maxB0 		= fitparams[8]
        
        # intialize N*M root system
        allRS = []
        for i in range(0, N):
                for j in range(0, M):
                        rs = pb.RootSystem()                              # create root system
                        rs.readParameters(path + name + ".xml")           # open plant and root parameter file
                        # set scale elongation function
                        for p in rs.getRootRandomParameter():
                                p.f_se = scale_elongation
                        p0 = rs.getRootRandomParameter(1)                 # get tap root parameters (root system type 1)                         
                        p1 = rs.getRootRandomParameter(2)                 # get first lateral root parameters (root system type 2)
                        srp = rs.getRootSystemParameter()                 # get plant parameter (maximum number of basal roots)

                        p0.lmax 	= lmax0
                        p0.theta 	= theta0
                        p0.r 		= r0
                        p1.ln 	        = ln1
                        p0.tropismN 	= tropismN0
                        p1.lb   	= lb1
                        p1.la 	        = la1
                        p1.lmax 	= lmax1
                        srp.maxB 	= maxB0

                        rs.setSeed(1)    
                        rs.getRootSystemParameter().seedPos = pb.Vector3d(-50 + distr * (i + 0.5), -distp/2 * M + distp * (j + 0.5), -3.)  # cm
                        rs.initialize(False)
                        allRS.append(rs)

        # Simulate
        time = 0
        dt   = 1
        while time < simtime:
                # update scales (from soil density) 
                scales = a*bulk_density**2 - b*bulk_density + c
                scale_elongation.data = scales
                
                for rs in allRS:
                        rs.simulate(dt)
                time += dt      

        # Export results as single vtp files (as polylines)
        ana = pb.SegmentAnalyser()                                                                     # see example 3b
        for z, rs in enumerate(allRS):
                # vtpname = "results/plantsb" + str(z) + ".vtp"
                # rs.write(vtpname)
                ana.addSegments(rs)  # collect all

        # Write all into single file (segments)
        # ana.write("results/plantsb_allopt.vtp")

        # Set periodic domain
        ana.mapPeriodic(distTr, distTp)       # 100x6 cm 
        ana.pack()                     
        # ana.write("results/plantsb_periodic2.vtp")
        # vp.plot_roots(ana, "length", "length (cm)")
                  
        rl_ = []

        for j in range(len(times)):
                ana.filter("creationTime", 0, times[j])
                rld = ana.distribution2("length", top, bot, left, right, n, m, True)      # 2D distribution of root length
                rl_.append(rld)

        # Reverse root length filtered by times (DAS) in ascending order
        rl_.reverse()                                                                 
        # Root length density obtained by dividing root length by the soil volume 
        rld_ =  np.array(rl_)/soilVolume      
        # Reshape simulated RLD to 2D array
        rld_sim = rld_.reshape(-1, rld_.shape[2])
        # Save data as a text file 
        np.savetxt("sim_dataopt_107days.txt", np.round(rld_sim, decimals=4), fmt='%.4f', delimiter=',')
        # RMSE
        err2 = np.sqrt(sum(sum((rld_sim - measured_RLD.values)**2))/measured_RLD.values.size)       
        
        print("RMSE at each grid: ", err2)
        return err2

# Define upper and lower bounds of each root parameter
# [lmax0,theta0,r0,ln1,tropismN0,lb1,la1,lmax1,maxB0]
bounds = [(100,250), (1.5, 2), (0, 3), (0, 3), (0, 5), (0, 2), (0, 5), (2,10), (4,8)]  

# Differential Evolution 
result = differential_evolution(err2, bounds,  strategy='best2bin', maxiter=1000, seed=3, popsize=20, mutation=0.5) 

print(result.x)


elapsed_time = (time.process_time() - t) / 3600

print("Time for program to execute in hours: ", elapsed_time)


# Optimized parameters

# lmax0  =     235.0961742
# theta0 =       1.82208357
# r0     =       2.72171692
# ln1  =         2.0889488
# tropismN0 =    1.36727873
# lb1    =       0.81725965
# la1    =       2.96734759
# lmax1  =       3.19606305
# maxB0  =       6.59418266

# t_sim  =      12.50
# rmse   =       0.94


