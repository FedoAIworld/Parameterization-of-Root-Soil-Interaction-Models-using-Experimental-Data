"""
Copyright: 2021
Author: Fernand B. Eloundou
-----------------------------
File: springbarley_rootsim.py
"""

# import libraries
import sys
sys.path.append("../../..") 
sys.path.append("../../../src/python_modules")
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import vtk_plot as vp
import plantbox as pb
import math
import time

# path to and name of plant and root parameter file
path = "../../../modelparameter/rootsystem/"
name = "spring_barley_opt"

"""
Simulate multiple root system architectures (RSA) with root development without any subsoil treatment. 

Simulate virtual soil profile wall (SPW) of 100 cm wide by 100 cm depth by scaling the elongation rate using
bulk density values based on the EquidistantGrid1D function in CPlantBox. 

Stimulate two plants (RSA) in 8 rows, and compute the RLD perpendicular to each row. 

Filter the RLD according to days after sowing as in the field experiment.
"""

t = time.process_time()

simtime = 107              # simulation time
N = 8                      # number of rows
M = 2                      # number of plants in a row 
distp = 3.                 # distance between the root systems along row[cm]
distr = 12.5               # distance between the rows[cm]
distTr = N*distr           # total row spacing
distTp = M*distp           # total distance between plants 
top = 0.                   # vertical top position (cm) (normally = 0)
bot = -100.                # vertical bot position (cm) (e.g. = -100 cm)
left = -47.5               # left along x-axis (cm)
right = 47.5               # right along x-axis (cm)
n = 20                     # number of layers, each with a height of (top-bot)/n
m = 20                     # number of horizontal grid elements (each with length of (right-left)/m)
exact = True               # calculates the intersection with the layer boundaries (true), only based on segment midpoints (false)

# soil volume
soilVolume =   5*5*5                # cm3

# time points during vegetation period in days
# (shooting=45,46) and (flowering= 71,73,74)
times = [74, 73, 71, 46, 45]     

# Define bulk densities 
scale_elongation = pb.EquidistantGrid1D(0, -120, 9)  			  # 0=surface; -120=depth; 9=number of layer boundries
bulk_density = np.array([1.30, 1.50, 1.56, 1.50, 1.53, 1.57, 1.59, 1.61])
a=2.52244967; b=9.477859748; c=9.058277729; 
scales = a*bulk_density**2 - b*bulk_density + c


scale_elongation.data = scales

# Make a root length distribution along the soil profile wall 

# Initializes N*M root systems

allRS = []
for i in range(0, N):
    for j in range(0, M):
        rs = pb.RootSystem()
        rs.readParameters(path + name + ".xml")
        for p in rs.getRootRandomParameter():
            p.f_se = scale_elongation
        rs.getRootSystemParameter().seedPos = pb.Vector3d(-50 + distr * (i + 0.5), -distp/2 * M + distp * (j + 0.5), -3.)  # cm
        # Create and set geometry
        box = pb.SDF_PlantBox (1000, 1000 , 1000)
        rs.setGeometry(box)
        rs.initialize(False)
        allRS.append(rs)
        
# Simulate
time = 0
dt   = 1
while time < simtime:
    # update scales (e.g. from water content, soil_strength) 
    scales = a*bulk_density**2 - b*bulk_density + c
    scale_elongation.data = scales
                
    for rs in allRS:
        rs.simulate(dt)
    time += dt   

# Export results as single vtp files (as polylines)
ana = pb.SegmentAnalyser()                           	# see example 3b
for z, rs in enumerate(allRS):
    vtpname = "results/plantsb" + str(z) + ".vtp"
    rs.write(vtpname)
    ana.addSegments(rs)  # collect all

# Write all into single file (segments)
ana.write("results/plantsb_allCTRLopt.vtp")

# Set periodic domain
ana.mapPeriodic(distTr, distTp)  
ana.pack()                        
ana.write("results/plantsb_periodicCTRLopt.vtp")
# vp.plot_roots(ana, "length", "length (cm)")
          
rl_ = []

for j in range(len(times)):
    ana.filter("creationTime", 0, times[j])
    rl = ana.distribution2("length", top, bot, -50, 50, n, m, True)
    rl_.append(rl)

rl_.reverse()                    				# reverse the root length of the order shooting (45, 46) and flowering (71, 73, 74)

rld_ =  np.array(rl_)/soilVolume

rld_simDL = rld_.reshape(-1, rld_.shape[2])                            # reshaping the RLD into 2D

np.savetxt("sim_data_ctrl.txt", np.round(rld_simDL, decimals=4), fmt='%.4f', delimiter=',')

# Shooting

rld = rld_[0]                                             	# day 45 during vegetation period
data_shooting1 = np.array([np.array(l) for l in rld])     

rld = rld_[1]                                            	# day 46 during vegetation period
data_shooting2 = np.array([np.array(l) for l in rld])     


# Flowering

rld = rld_[2]							                    # day 71 during vegetation period
data_flowering1 = np.array([np.array(l) for l in rld]) 

rld = rld_[3]							                    # day 73 during vegetation period
data_flowering2 = np.array([np.array(l) for l in rld]) 

rld = rld_[4]     						                    # day 74 during vegetation period                                      
data_flowering3 = np.array([np.array(l) for l in rld])     

    

x = np.linspace(-47.5, 47.5, 20)
y = np.linspace(-2.5, -97.5, 20)
levels = np.linspace(0, 4, 21)
X, Y = np.meshgrid(x, y)


# Plot RLD during shooting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
fig.suptitle("RLD During Plant Shooting", fontsize=16, y=1.07)

fig12 = ax1.contourf(X, Y, data_shooting1, cmap='YlGnBu', levels=levels)
ax1.set_xlabel("Distance of the row [cm]", fontsize=15, labelpad=15)
ax1.set_ylabel("Depth [cm]", fontsize=15)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position("top")
cbar = fig.colorbar(fig12, ax=ax1)

fig13 = ax2.contourf(X, Y, data_shooting2, cmap='YlGnBu', levels=levels)
ax2.set_xlabel("Distance of the row [cm]", fontsize=15, labelpad=15)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position("top")
cbar = fig.colorbar(fig13, ax=ax2)
cbar.ax.set_ylabel("Root Length Density [cm cm$^-$$^3$]", fontsize=15, labelpad=15)

# plt.savefig("RLD during shooting CTRL.png")
plt.show()

# Plot RLD during flowering
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
fig2.suptitle("RLD During Plant Flowering", fontsize=16, y=1.07)

fig21 = ax1.contourf(X, Y, data_flowering1, cmap='YlGnBu', levels=levels)
ax1.set_xlabel("Distance of the row [cm]", fontsize=15, labelpad=15)
ax1.set_ylabel("Depth [cm]", fontsize=15)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position("top")
cbar1 = fig2.colorbar(fig21, ax=ax1)

fig22 = ax2.contourf(X, Y, data_flowering2, cmap='YlGnBu', levels=levels)
ax2.set_xlabel("Distance of the row [cm]", fontsize=15, labelpad=15)
ax2.xaxis.tick_top()
ax2.xaxis.set_label_position("top")
cbar1 = fig2.colorbar(fig22, ax=ax2)

fig23 = ax3.contourf(X, Y, data_flowering3, cmap='YlGnBu', levels=levels)
ax3.set_xlabel("Distance of the row [cm]", fontsize=15, labelpad=15)
ax3.xaxis.tick_top()
ax3.xaxis.set_label_position("top")
cbar1 = fig2.colorbar(fig23, ax=ax3)

cbar1.ax.set_ylabel("Root Length Density [cm cm$^-$$^3$]", fontsize=15, labelpad=15)

# plt.savefig("RLD during flowering CTRL.png")
plt.show()

elapsed_time = time.process_time() - t

print("Time for program to execute in seconds: ", elapsed_time)
