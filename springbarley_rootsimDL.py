"""
Copyright: 2021
Authors: Fernand B. Eloundou
         Dr. Daniel Leitner 
-------------------------------
File: springbarley_rootsimDL.py
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
import faulthandler; faulthandler.enable()
from pyevtk.hl import gridToVTK                                 # pip3 install pyevtk

# path to and name of plant and root parameter file
path = "../../../modelparameter/rootsystem/"
name = "spring_barley_opt"

"""
Simulate multiple root system architectures (RSA) with root development affected by deep loosening (DL).

Simulate virtual soil profile wall (SPW) of 300 cm wide by 100 cm depth with a slit (DL) of 30 cm placed in the middle of the SPW. 
The elongation rate is scaled using bulk density values both outside and inside the slit based on the EquidistantGrid3D function in CPlantBox. 
Simulate two plants (RSA) in 24 rows, and compute the RLD perpendicular to each row for the simulation time. 

Filter RLD according to days after sowing as in the field experiment.
"""

simtime = 107                           # simulation time
N       = 24  						    # number of rows, 8 per [m]
M       = 2  						    # number of plants in a row 
distp   = 3.  				            # distance between the root systems along row [cm]
distr   = 12.5  						# distance between the rows [cm]
distTr  = N * distr  					# total row spacing
distTp  = M * distp  					# total distance between plants 
top     = 0.  						    # vertical top position (cm) (normally = 0)
bot     = -100.  						# vertical bot position (cm) (e.g. = -100 cm)
left    = -150  					    # left along y-axis (cm)
right   = 150  					        # right along y-axis (cm)
n       = 20  						    # number of layers, each with a height of (top-bot)/n
m       = 60  						    # number of horizontal grid elements (each with length of (right-left)/m)
m2      = int(M * distp)  		        # resolution of 1 cm  
exact   = True  						# calculates the intersection with the layer boundaries (true), only based on segment midpoints (false)

# soil volume
soilVolume = 5 * 5 * 5

# Time points during vegetation period in days
# (shooting=45,46) and (flowering= 71,73,74)
times = [74, 73, 71, 46, 45]  			 
 

# bulk density grid definition
y_left = -200
y_right = 200
distp_row = 6
z_top = 0.
z_bot = -200.
horizontal_grid_num = 200
y_distance = 6
num_layers = 40

# Define bulk denistitis
print("EquidistantGrid3D: ", y_right-y_left, distp_row, z_top-z_bot, "cm, res:", horizontal_grid_num, y_distance, num_layers)
scale_elongation = pb.EquidistantGrid3D(y_right-y_left, distp_row, z_top-z_bot, horizontal_grid_num, y_distance, num_layers)     # grid is centered in x, y 
bulk_density = np.ones((horizontal_grid_num, y_distance, num_layers))


# depth of 200 cm and 40 layers of 5 cm distance
depth = np.linspace(5, 200, 40)                                        
# list of the position of layers
zi_list = []                                                           

for i in depth:
    zi = num_layers - int(i / ((z_top-z_bot) / num_layers))            # gridz is from [-depth - 0], i.e. surface is at end of array
    zi_list.append(zi)  									            

print("z indices", zi_list) 

# bulk densities of the field out of the slit across depths 
BD = [1.30, 1.30, 1.30, 1.30, 1.50, 1.50, 1.56, 1.56, \
      1.50, 1.50, 1.53, 1.53, 1.57, 1.57, 1.59, 1.59, 1.59, \
      1.59, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61, \
      1.61, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61, 1.61, \
      1.61, 1.61, 1.61, 1.61, 1.61]

bulk_density[:,:, zi_list[0]:] = 1.30                                  # bulk density on the surface of the soil outside the slit
count = 0
for i in BD:
    bulk_density[:,:, zi_list[count+1]:zi_list[count]] = i
    count+=1                   

# # slit only                                                          # definition of slit position 30 cm horizontally 
xi1 = int(185/((y_right-y_left) / horizontal_grid_num))  			   # -15 cm
xi2 = int(215/((y_right-y_left) / horizontal_grid_num))  			   # 15 cm

print("x indices", xi1, xi2)

# bulk densities in the slit
BD_slit = [1.20, 1.20, 1.20, 1.20, 1.32, 1.32, 1.33, 1.33, \
            1.47, 1.47, 1.59, 1.59, 1.58, 1.58, 1.55, 1.55, 1.55, \
            1.55, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, \
            1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, 1.58, \
            1.58, 1.58, 1.58, 1.58, 1.58]

bulk_density[xi1:xi2,:, zi_list[0]:] = 1.20                             # bulk density on the surface of the soil in the slit
count = 0
for j in BD_slit:
    bulk_density[xi1:xi2,:, zi_list[count+1]:zi_list[count]] = j
    count+=1

# Scaling function using elongation rates from Morandage et al.
a=2.52244967; b=9.477859748; c=9.058277729; 
scales = a*bulk_density**2 - b*bulk_density + c 					    
scale_elongation.data = scales.flatten('F')  						    # set proportionality factor



# vizualise scalar grid
X = np.linspace(y_left, y_right, horizontal_grid_num)
Y = np.linspace(-distp_row, +distp_row, y_distance)
Z = np.linspace(z_bot, z_top, num_layers)
# gridToVTK("results/bulk_density", X, Y, Z, pointData={"bulk_density": bulk_density, "scales ": scales.reshape(horizontal_grid_num, y_distance, num_layers)})               # "scales ": scales.reshape(m, m2, n)})

# Make a root length distribution along the soil profile wall
# Initializes N*M root systems
allRS = []
for i in range(0, N):
    for j in range(0, M):
        rs = pb.RootSystem()                                                     # create root system
        rs.readParameters(path + name + ".xml")                                  # open plant and root parameter file
        for p in rs.getRootRandomParameter():  								     # set scale elongation function for all root types
            p.f_se = scale_elongation          
        rs.getRootSystemParameter().seedPos = pb.Vector3d(left + distr * (i + 0.5), -distp / 2 * M + distp * (j + 0.5), -3.)  # cm
        
        # Create and set geometry
        box = pb.SDF_PlantBox (1000, 1000, 1000)
        rs.setGeometry(box)
        rs.initialize(False)
        allRS.append(rs)
 

# Simulate
time = 0
dt = 1                                                                                  # day
while time < simtime:                                                                   # in the future coupling with dynamic water movement 
    print("day", time)
    
    # update scales (from soil density)
    scales = a*bulk_density**2 - b*bulk_density + c							            
    scale_elongation.data = scales.flatten('F')
        
    for rs in allRS:
        rs.simulate(dt)
    time += dt


# Export results as single vtp files (as polylines)
ana = pb.SegmentAnalyser()                                                              
for z, rs in enumerate(allRS):
    # vtpname = "results/plantsb" + str(z) + ".vtp"
    # rs.write(vtpname)
    ana.addSegments(rs)  											                    # collect all

# Write all into single file (segments)
ana.write("results/plantsb_allDL.vtp")

# Set periodic domain
ana.mapPeriodic(distTr, distTp)     
ana.pack()                      
ana.write("results/plantsb_periodicDL.vtp")

# vp.plot_roots(ana, "length", "length (c)")

rl_ = []
for j in range(len(times)):
    # print("creating rld for time", times[j])
    ana.filter("creationTime", 0, times[j])                                  # filter root length by times in descending order
    rld = ana.distribution2("length", top, bot, left, right, n, m, False)    # 2D distribution of root length
    rl_.append(rld)

# reverse root length of the order shooting (45, 46) and flowering (71, 72, 73)   
rl_.reverse()                                                          
# root length density
rld_ = np.array(rl_) / soilVolume
# reshape simulated RLD to 2D array
rld_simDL = rld_.reshape(-1, rld_.shape[2])                            # reshaping the RLD into 2D
# save simulated RLD as a text file
np.savetxt("sim_data_DL.txt", np.round(rld_simDL, decimals=4), fmt='%.4f', delimiter=',')


# Shooting

data_shooting1 = rld_simDL[:20, 30:50] 							        # day 45 during vegetation period                                      
     
data_shooting2 = rld_simDL[20:40, 30:50]  								# day 46 during vegetation period


# Flowering

data_flowering1 = rld_simDL[40:60, 30:50]  								# day 71 during vegetation period
   
data_flowering2 = rld_simDL[60:80, 30:50]  								# day 73 during vegetation period
     
data_flowering3 = rld_simDL[80:100, 30:50]  							# day 74 during vegetation period

# definition of contour lines in 2D
x = np.linspace(0, 95, 20)  									        # x features along distance of the row
y = np.linspace(-2.5, -97.5, 20)  									    # y features
levels = np.linspace(0, 4, 21)                                          # number and positions of contour lines/regions
X, Y = np.meshgrid(x, y)                                                # height values over which the contour is drawn


# Plot during shooting
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

# plt.savefig("RLD during shooting DL.png")
plt.show()

# Plot during flowering
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

# plt.savefig("RLD during flowering DL.png")
plt.show()

