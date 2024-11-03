import random
import numpy as np
import pyshtools as shp_harm
import matplotlib.pyplot as plt
import time
import cvxpy as cvx
from SparsePlacePSO import SparsePlaceSpherical, SparsePlaceRobust
from functools import *

def compose(*fs):
    def comp(f, g):
        return lambda x : f(g(x))       
    return reduce(comp, fs, lambda x : x)

def clear_stuff(f, *args, **kwargs):
    plt.close()
    plt.cla()
    plt.clf()
    return f(*args, **kwargs)

## Cartesian to Spherical Coordinate
def cartesian2spherical(x,y, r_head):
    z = np.sqrt(r_head**2-x**2-y**2)
    theta_loc = np.arccos(z / r_head)
    theta_loc = np.pi / 2 - theta_loc

    phi_loc = np.zeros(len(x))
    for i in range(len(x)):
        if np.abs(x[i]) < 1e-7:
            if y[i] >= 0:
                phi_loc[i] = 0
            else:
                phi_loc[i] = np.pi
        elif np.abs(y[i]) < 1e-7:
            if y[i] >= 0:
                phi_loc[i] = np.pi / 2
            else:
                phi_loc[i] = 3 * np.pi / 2
        elif x[i] > 0 and y[i] > 0:
            phi_loc[i] = np.arctan(y[i] / x[i])
        elif x[i] < 0 and y[i] > 0:

            phi_loc[i] = np.pi - np.arctan(y[i] / (-1 * x[i]))

        elif x[i] < 0 and y[i] < 0:
            phi_loc[i] = np.pi + np.arctan(y[i] / x[i])
        elif x[i] > 0 and y[i] < 0:
            phi_loc[i] = 2 * np.pi - np.arctan((-1 * y[i]) / x[i])

    return theta_loc, phi_loc


# locations for focus and cancel matrices
save_title_focus = "Forward_Model/Focus_Matrix_SparsePlace/Forward_Focus_IRB_20mAcm^2_6cmPatch_neg.npy"
save_title_cancel = "Forward_Model/Cancel_Matrix_SparsePlace/Forward_Cancel_IRB_20mAcm^2_6cmPatch_neg.npy"


######### Specify Head Model ###########

## Nunez and Srinivasan 2006 Book
radius_head = 9.2 # cm
thickness_skull = 0.5 #cm
thickness_scalp = 0.6 #cm
thickness_CSF = 0.1 #cm

no_of_layers = 4

## Conductivity
skull_cond = 0.006 #S/m
brain_cond = 0.33 #S/m
CSF_cond = 1.79 #S/m
scalp_cond = 0.3 #S/m

####################### IRB Simulation ##############################
radius_vec = np.array([radius_head-thickness_skull-thickness_CSF, radius_head-thickness_skull, radius_head])
cond_vec = np.array([brain_cond, CSF_cond, skull_cond])
####################################################################################

L = 500

######### TDCS ##########################
patch_radius_TDCS = 0.72 #cm
pos = [9.2-1.3, np.pi/2.0, 0]

inj_curr = 200/(np.pi*patch_radius_TDCS**2)

######### SparsePlace ##########################

custom_grid = False

## Patch Specifications
patch_size = 6 #cm
elec_radius = 0.5
elec_spacing = 1.5

## Cancel Points
offset = 0.5
thickness = 2
spacing = 0.2

## Current Density
Jdes = 16.8 / 10 #mA/cm^2

# Electrode pattern
#          -
#     -         -
#          +
#     -         -
#          -
J = np.zeros(37,)
J[0] = 1 ## mA
for i in range(1,7):
    J[[i]] = -1.0 / 6 ## mA

focus_depth = 1.3
focus_point = np.array([radius_vec[-1]-focus_depth, np.pi/2.0, 0])
focus_points = focus_point.reshape(1, -1)

centre = focus_point.reshape((-1,))
default_model = SparsePlaceSpherical(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
    elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
    custom_grid=custom_grid, theta_elec=None, phi_elec=None)
cancel_points = default_model.uniform_ring_sample(centre, offset, thickness)

# for hinge place, jtol ~= 0.7-0.8 * Jdes
Jsafety = np.array((4,25))*4/(np.pi*elec_radius**2)
Jsafety = 100 / (np.pi * elec_radius**2)

beta = 37*Jsafety
