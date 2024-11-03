from functools import reduce
import random
import numpy as np
import pyshtools as shp_harm
import matplotlib.pyplot as plt
import time
import cvxpy as cvx
from SparsePlace import SparsePlaceSpherical, SparsePlaceRobust

######### Specify Head Model ###########

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

## Saving
# if 'save' in __dict__:
# save_title_focus = "Forward_Model/Focus_Matrix_SparsePlace/Forward_Focus_IRB_20mAcm^2_6cmPatch_neg.npy"
# save_title_cancel = "Forward_Model/Cancel_Matrix_SparsePlace/Forward_Cancel_IRB_20mAcm^2_6cmPatch_neg.npy"
# plot = False
# Af = None
# Ac = None

## Loading Af and Ac
Af = np.load("Forward_Model/Focus_Matrix_SparsePlace/Forward_Focus_IRB_20mAcm^2_6cmPatch_neg.npy")
Ac = np.load("Forward_Model/Cancel_Matrix_SparsePlace/Forward_Cancel_IRB_20mAcm^2_6cmPatch_neg.npy")


### Current Density
Jdes = 16.8 / 10 #mA/cm^2


## Target Point
focus_depth_str = input("How deep focus pt? (enter for default = 1.3)")
try:
    focus_depth = float(focus_depth_str)
except:
    focus_depth = 1.3
focus_point = np.array([radius_head-focus_depth, np.pi/2.0, 0])

### Vanilla SparsePlace

####### Generating Cancel Points
centre = focus_point.reshape((-1,))



def compose(*fs):
    def comp(f, g):
        return lambda x : f(g(x))       
    return reduce(comp, fs, lambda x : x)


def clear_stuff(f, *args, **kwargs):
    plt.close()
    plt.cla()
    plt.clf()
    return f(*args, **kwargs)


def make_model(pert=0):
    cond = input('Condvec (press enter for default: brain 0.33, ' +
        'csf 1.79, skull 0.006)\n')
    if cond == '':
        condvec = cond_vec
    else:
        print('here')
        condvec = list(map(lambda s : float(s.strip()), cond.split(',')))
        print(condvec)
        cond_diffs = pert * np.array([random.choice((-1, 1)) for _ in range(len(condvec))])
        condvec += condvec * cond_diffs
        print(condvec)
    # radius_head = 9.2 # cm, thickness_skull = 0.5 #cm thickness_scalp = 0.6 #cm thickness_CSF = 0.1 #cm
    rad = input('Radii (default 9.2, 0.5, 0.6, 0.1)\n')
    if rad == '':
        radvec = radius_vec
    else:
        print('here2')
        rad_in = list(map(lambda s : float(s.strip()), rad.split(',')))
        print('og radii:', [rad_in[0]-rad_in[1]-rad_in[3], rad_in[0]-rad_in[1], rad_in[0]])
        radvec_diffs = pert * np.array([random.choice((-1, 1)) for _ in range(len(rad_in))])
        rad_in_perturb = rad_in * (1 + radvec_diffs)
        assert(sum(rad_in_perturb[1:]) < rad_in_perturb[0])
        radius_head, thickness_skull, thickness_scalp, thickness_CSF = rad_in_perturb
        radvec = np.array([radius_head-thickness_skull-thickness_CSF, radius_head-thickness_skull, radius_head])
        print('Radii:', radvec)

    sparsePlace = SparsePlaceSpherical(cond_vec=condvec, radius_vec=radvec, patch_size=patch_size,
        elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing, custom_grid=custom_grid,
        theta_elec=None, phi_elec=None)
    cancel_points = sparsePlace.uniform_ring_sample(centre, offset, thickness)
    return sparsePlace, cancel_points


# Electrode pattern
J = np.zeros(37,)
J[0] = 1 ## mA
for i in range(1,7):
    J[[i]] = -1.0 / 6 ## mA

def make_fns(pert=0):
    sparsePlace, cancel_points = make_model(pert=pert)
    elec_pat = lambda gran=20 : clear_stuff(sparsePlace.plot_elec_pttrn, J=J,x_lim=np.array((-5,5)), y_lim=np.array((-5,5)), disc_granularity=gran, fname="Figures/Example_Figs/elecpat.png")
    curr_dens = lambda : sparsePlace.plot_curr_density(r=focus_point[0], J=J, x_limit=np.array([-5,5]), y_limit=np.array([-5,5]), abs=False)
    volt = lambda : sparsePlace.plot_voltage(r=focus_point[0], J=J, x_limit=np.array([-5,5]), y_limit=np.array([-5,5]), abs=False)
    return sparsePlace, cancel_points, elec_pat, curr_dens, volt

focus_points = np.asarray([focus_point])
sparsePlace, cancel_points, elec_pat, curr_dens, volt = make_fns()
# sparsePlace.area_above_x
# # for hinge place, jtol ~= 0.7-0.8 * Jdes
Jsafety = np.array((4,25))*4/(np.pi*elec_radius**2)
# mix_per_lst = 0.8
beta = 37*Jsafety


##### Calculating Forward Matrices
Af, Ac = sparsePlace.forward_model(cancel_points=cancel_points, focus_points=focus_point.reshape(1,-1), print_elec_pattern=True, save=True, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel)



v = lambda : sparsePlace(Af=Af, Ac=Ac, per=0.75, optimizer=1, Jdes=Jdes, Jsafety=2000 / (np.pi * elec_radius**2),
    focus_points=np.array([focus_point]), cancel_points=cancel_points)

# J_hinge_60, area_HP = sparsePlace(per=0.75, optimizer=2, Af=Af, Ac=Ac, Jtol=Jdes*0.6, Jdes=Jdes, Jsafety=Jsafety, focus_points=focus_point.reshape(-1,3), beta=beta, print_elec_pattern=False, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, density_thresh=Jdes*0.75, fname_density='Figures/IRB/HP_Density_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_area='Figures/IRB/HP_Area_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_elec='Figures/IRB/HP_ElecPttrn_'+str(int(Jsafety))+"_6cmPatch_neg")
# vol_HP = sparsePlace.volume_above_x(r=radius_head-1.3, density_thresh=Jdes*0.75, per=0.75, dr=0.01, plot=False, J=J_hinge_60)
# J_sparse, area_SP = sparsePlace(per=0.75, optimizer=1, Af=Af, Ac=Ac, Jdes=Jdes, Jsafety=Jsafety, focus_points=focus_point.reshape(-1,3), beta=beta, print_elec_pattern=False, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, density_thresh=Jdes*0.75, fname_density='Figures/IRB/SP_Density_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_area='Figures/IRB/SP_Area_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_elec='Figures/IRB/SP_ElecPttrn_'+str(int(Jsafety))+"_6cmPatch_neg")
# vol_SP = sparsePlace.volume_above_x(r=radius_head-1.3, density_thresh=0.75*Jdes, per=0.75, dr=0.01, plot=False, J=J_sparse)

