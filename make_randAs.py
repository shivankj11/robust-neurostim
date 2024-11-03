from modelinfo import *



def make_model(model=SparsePlaceSpherical, basic=True, pert=0):

    if basic:
        sparsePlace = model(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
        elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing, custom_grid=custom_grid,
        theta_elec=None, phi_elec=None)
        focus_point = np.array([radius_head-focus_depth, np.pi/2.0, 0])
        ####### Generating Cancel Points
        centre = focus_point.reshape((-1,))
        cancel_points = sparsePlace.uniform_ring_sample(centre, offset, thickness)
        return sparsePlace, focus_point, cancel_points
    
    ## Target Point
    focus_depth_str = input("How deep focus pt? (enter for default = 1.3)")
    try:
        focus_depthp = float(focus_depth_str)
    except:
        focus_depthp = 1.3
    focus_point = np.array([radius_vec[-1]-focus_depthp, np.pi/2.0, 0])

    # custom conductivities
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
        radius_head2, thickness_skullp, thickness_scalpp, thickness_CSFp = rad_in_perturb
        radvec = np.array([radius_head2-thickness_skullp-thickness_CSFp, radius_head2-thickness_skullp, radius_head2])
        print('Radii:', radvec)

    sparsePlace = model(cond_vec=condvec, radius_vec=radvec, patch_size=patch_size,
        elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
        custom_grid=custom_grid, theta_elec=None, phi_elec=None)
    centre = focus_point.reshape((-1,))
    cancel_points = sparsePlace.uniform_ring_sample(centre, offset, thickness)
    return sparsePlace, focus_point, cancel_points



( # params
# radius_head = 9.2 # cm
# thickness_skull = 0.5 #cm
# thickness_scalp = 0.6 #cm
# thickness_CSF = 0.1 #cm
# no_of_layers = 4
# skull_cond = 0.006 #S/m
# brain_cond = 0.33 #S/m
# CSF_cond = 1.79 #S/m
# scalp_cond = 0.3 #S/m
# radius_vec = np.array([radius_head-thickness_skull-thickness_CSF, radius_head-thickness_skull, radius_head])
# cond_vec = np.array([brain_cond, CSF_cond, skull_cond])
# patch_radius_TDCS = 0.72 #cm
# pos = [9.2-1.3, np.pi/2.0, 0]
# inj_curr = 200/(np.pi*patch_radius_TDCS**2)
# patch_size = 6 #cm
# elec_radius = 0.5
# elec_spacing = 1.5
# ## Cancel Points
# offset = 0.5
# thickness = 2
# spacing = 0.2
# Jdes = 16.8 / 10 #mA/cm^2
# Jsafety = 2000 / (np.pi * elec_radius**2)
)
# focus_depth += 0.5

### Vanilla SparsePlace
sparsePlace, focus_point, cancel_points = make_model(model=SparsePlaceSpherical)
sparsePlaceRobust, focus_point, cancel_points = make_model(model=SparsePlaceRobust)

# plots
elec_pat = lambda m, gran=20 : clear_stuff(m.plot_elec_pttrn, J=J, x_lim=np.array((-5,5)),
    y_lim=np.array((-5,5)), disc_granularity=gran, fname="Figures/Example_Figs/elecpat.png")
curr_dens = lambda m : m.plot_curr_density(r=focus_point[0], J=J, x_limit=np.array([-5,5]),
    y_limit=np.array([-5,5]), abs=False)
volt = lambda m : m.plot_voltage(r=focus_point[0], J=J, x_limit=np.array([-5,5]),
    y_limit=np.array([-5,5]), abs=False)

# sparsePlace.area_above_x
# mix_per_lst = 0.8


##### Calculating Forward Matrices

## Run forward model and save Af, Ac
# locations for focus and cancel matrices
save_title_focus = "Forward_Model/Focus_Matrix_SparsePlace/Forward_Focus_IRB_20mAcm^2_6cmPatch_neg.npy"
save_title_cancel = "Forward_Model/Cancel_Matrix_SparsePlace/Forward_Cancel_IRB_20mAcm^2_6cmPatch_neg.npy"
# # Af, Ac = None, None
# Af, Ac = sparsePlace.forward_model(cancel_points=cancel_points, focus_points=focus_points,
#     print_elec_pattern=False, save=True, save_title_focus=save_title_focus,
#     save_title_cancel=save_title_cancel)

# sparsePlace.SparsePlaceHinge(Jdes=Jdes, Jsafety=Jsafety, Jtol=0.8*Jdes)
## Load Af and Ac
Af = np.load(save_title_focus)
Ac = np.load(save_title_cancel)


v = (lambda m, per=0.7, Af=Af, Ac=Ac, focus_points=focus_points, cancel_points=cancel_points, 
    Jdes=Jdes, Jsafety=Jsafety, **kwargs : m(Af=Af, Ac=Ac, per=per, optimizer=1, Jdes=Jdes, 
    Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False, **kwargs))

# res = v (m=sparsePlace)

# res2 = v (m=sparsePlaceRobust)

def generate_randomAs():
    N = 40
    Afs = None #np.zeros((N, *Af.shape))
    Acs = None #np.zeros((N, *Ac.shape))

    for i in range(N):
        if i >= 20: exit(0)
        condvec = np.random.uniform(low=(0.95 * cond_vec), high=(1.05 * cond_vec))
        rvs = np.array([9.2, 0.5, 0.1])
        radius_head, thickness_skull, thickness_CSF = np.random.uniform(low=(0.95 * rvs), high=(1.05 * rvs))
        radvec = np.array([radius_head-thickness_skull-thickness_CSF, radius_head-thickness_skull, radius_head])
        if (thickness_skull+thickness_CSF) > focus_depth:
            i -= 1
            continue
        S = SparsePlaceSpherical(cond_vec=condvec, radius_vec=radvec, patch_size=patch_size,
                elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
                custom_grid=custom_grid, theta_elec=None, phi_elec=None)
        Afr, Acr = S.forward_model(focus_points, cancel_points)
        Afs = np.load("afs_rand.npy")
        Acs = np.load("acs_rand.npy")
        Afs[i] = Afr
        np.save("afs_rand.npy", Afs)
        Acs[i] = Acr
        np.save("acs_rand.npy", Acs)
        print(str(i) + '%', end='\n\n')

generate_randomAs()

# print(res[0])
# print('Robust:')
# print(res2[0])
# J_hinge_60, area_HP = sparsePlace(per=0.75, optimizer=2, Af=Af, Ac=Ac, Jtol=Jdes*0.6, Jdes=Jdes, Jsafety=Jsafety, focus_points=focus_point.reshape(-1,3), beta=beta, print_elec_pattern=False, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, density_thresh=Jdes*0.75, fname_density='Figures/IRB/HP_Density_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_area='Figures/IRB/HP_Area_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_elec='Figures/IRB/HP_ElecPttrn_'+str(int(Jsafety))+"_6cmPatch_neg")
# vol_HP = sparsePlace.volume_above_x(r=radius_head-1.3, density_thresh=Jdes*0.75, per=0.75, dr=0.01, plot=False, J=J_hinge_60)
# J_sparse, area_SP = sparsePlace(per=0.75, optimizer=1, Af=Af, Ac=Ac, Jdes=Jdes, Jsafety=Jsafety, focus_points=focus_point.reshape(-1,3), beta=beta, print_elec_pattern=False, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, density_thresh=Jdes*0.75, fname_density='Figures/IRB/SP_Density_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_area='Figures/IRB/SP_Area_'+str(int(Jsafety))+"_6cmPatch_neg.png", fname_elec='Figures/IRB/SP_ElecPttrn_'+str(int(Jsafety))+"_6cmPatch_neg")
# vol_SP = sparsePlace.volume_above_x(r=radius_head-1.3, density_thresh=0.75*Jdes, per=0.75, dr=0.01, plot=False, J=J_sparse)
