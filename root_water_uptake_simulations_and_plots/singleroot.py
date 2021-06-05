"""
Author: Deepanshu Khare
-----------------------
File: singleroot.py
"""

import os
import matplotlib.pyplot as plt
from vtk_tools import *
import van_genuchten as vg
import math
import numpy as np

name = "singleroot_Honly"  # this name should be unique
suffix = "_MS_DL"

"""
Simulation of transpiration rates or root water uptake
Generates the actual transpiration rates used in RWU_simulation_plots.ipynb file
"""
# go to the right place
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)
os.chdir("../../../build-cmake/rosi_benchmarking/coupled_1p_richards")

# run simulation
os.system("./coupled_periodic input/" + name + ".input")

# move results to folder 'name'
if not os.path.exists("results_" + name + suffix):
    os.mkdir("results_" + name + suffix)
os.system("mv " + name + "* " + "results_" + name + suffix + "/")
os.system("cp input/" + name + ".input " + "results_" + name + suffix + "/")

# 0 time [s], 1 actual transpiration [kg/s], 2 potential transpiration [kg/s], 3 maximal transpiration [kg/s],
# 4 collar pressure [Pa], 5 calculated actual transpiration, 6 time [s]
with open("results_" + name + suffix + "/" + name + "_actual_transpiration.txt", 'r') as f:  # benchmarkC12c_actual_transpiration. or benchmarkC12bc_actual_transpiration
    d = np.loadtxt(f, delimiter = ',')
c = 24 * 3600  #  [kg/s] -> [kg/day]
t = d[:, 0] / c  # [s] -> [day]

print("potential", d[-1, 2] * c)
print("actual", d[-1, 1] * c)
print("actual", d[-1, 5] / 1000)  							# Strange behaviour of simplistically calculated radial flows

""" Plot transpiration """
fig, ax1 = plt.subplots()
ax1.plot(t, 1000 * d[:, 2] * c, 'k')  						# potential transpiration
ax1.plot(t, 1000 * d[:, 1] * c, 'r-,')  						# actual transpiration

ax1b = ax1.twinx()
ctrans = np.cumsum(np.multiply(1000 * d[1:, 1] * c, (t[1:] - t[:-1])))
ax1b.plot(t[1:], ctrans, 'c--', color = 'blue')  					# cumulative transpiration (neumann)
ax1b.tick_params(axis= 'y', labelcolor = 'b')
ax1.set_xlabel("Time [days]")
ax1.set_ylabel("Transpiration [mL/day]")
ax1b.set_ylabel("Cumulative transpiration $[mL]$", color = "b")
ax1.legend(["potential", "actual", "cumulative"], loc = 'upper left')
plt.savefig("results_" + name + suffix + "/" + name + suffix + '_transpiration.png', dpi=300, bbox_inches='tight')

""" Plot root collar pressure """
fig, ax2 = plt.subplots()
ax2.plot(t, (d[:, 4] - 1.e5) * 100. / 1.e3 / 9.81, 'r-')  				# root collar pressure head (convert from Pa to Head)  
ax2.set_xlabel("Time [days]")
ax2.set_ylabel("Pressure at root collar [cm]")
plt.savefig("results_" + name + suffix + "/" + name + suffix + '_collarPressure.png', dpi=300, bbox_inches='tight')

plt.show()
