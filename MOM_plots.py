from modelinfo import *


save_title_focus = "Forward_Model/Focus_Matrix_SparsePlace/Forward_Focus_IRB_20mAcm^2_6cmPatch_neg.npy"
save_title_cancel = "Forward_Model/Cancel_Matrix_SparsePlace/Forward_Cancel_IRB_20mAcm^2_6cmPatch_neg.npy"

Af = np.load(save_title_focus)
Ac = np.load(save_title_cancel)
per = 0.7
focusr = np.min(focus_points[:, 0])

# original model on original A
S = SparsePlaceSpherical(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
    elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
    custom_grid=custom_grid, theta_elec=None, phi_elec=None)
Iog, aaxSPog = S(Af=Af, Ac=Ac, per=per, optimizer=1, Jdes=Jdes, 
    Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)


elec_pat = lambda gran=250 : clear_stuff(S.plot_elec_pttrn, J=J,x_lim=np.array((-5,5)), y_lim=np.array((-5,5)), disc_granularity=gran, fname="Figures/Example_Figs/elecpat.png")
curr_dens = lambda : S.plot_curr_density(r=focus_point[0], J=J, x_limit=np.array([-5,5]), y_limit=np.array([-5,5]), abs=False)
volt = lambda : S.plot_voltage(r=focus_point[0], J=J, x_limit=np.array([-5,5]), y_limit=np.array([-5,5]), abs=False)

from results import Iog, Irob
plot1 = lambda : S.plot_elec_pttrn(J=Iog,x_lim=np.array((-5,5)), y_lim=np.array((-5,5)), disc_granularity=300, fname="Figures/Example_Figs/elecpat_og.png")
plot2 = lambda : S.plot_elec_pttrn(J=Irob,x_lim=np.array((-5,5)), y_lim=np.array((-5,5)), disc_granularity=300, fname="Figures/Example_Figs/elecpat_rob.png")
plot3 = lambda : S.plot_elec_pttrn(J=Iog-Irob,x_lim=np.array((-5,5)), y_lim=np.array((-5,5)), disc_granularity=300, fname="Figures/Example_Figs/elecpat_diff.png")
