from modelinfo import *

save_title_focus = "Forward_Model/Focus_Matrix_SparsePlace/Forward_Focus_IRB_20mAcm^2_6cmPatch_neg.npy"
save_title_cancel = "Forward_Model/Cancel_Matrix_SparsePlace/Forward_Cancel_IRB_20mAcm^2_6cmPatch_neg.npy"

Af = np.load(save_title_focus)
Ac = np.load(save_title_cancel)

SPargs = {'radius_vec' : radius_vec, 'patch_size' : patch_size,
    'elec_radius' : elec_radius, 'elec_spacing' : elec_spacing, 'max_l' : L, 'spacing' : spacing,
    'custom_grid' : custom_grid, 'theta_elec' : None, 'phi_elec' : None}

M = SparsePlaceRobust(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
    elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
    custom_grid=custom_grid, theta_elec=None, phi_elec=None)

per = 0.7
Jrobust = M.SparsePlacePSO(Af=Af, Ac=Ac, per=per, optimizer=4, Jdes=Jdes, 
    Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False, radius_vec=radius_vec, patch_size=patch_size,
    elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
    custom_grid=custom_grid, theta_elec=None, phi_elec=None, SPargs=SPargs)

np.save("Jrobust.npy", Jrobust)