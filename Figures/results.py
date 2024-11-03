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


# original model on random As
randConds = np.load("randConds.npy")
randRads = np.load("randRads.npy")
Afrs2 = np.load("Afrs2.npy") # DO match
Acrs2 = np.load("Acrs2.npy")
Afs = np.load("afs_rand.npy") # DONT match above vecs
Acs = np.load("acs_rand.npy")

aaxSPrand = np.load("aaxSPrand_new1.npy")

# worst case model on original A
WC = SparsePlaceRobust(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
    elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
    custom_grid=custom_grid, theta_elec=None, phi_elec=None)

Iwc, aaxWCog = WC(Af=Af, Ac=Ac, per=per, optimizer=1, Jdes=Jdes, 
    Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)
aaxWCog = S.area_above_x(per, focusr, J=Iwc)


# worst case model on random As
aaxWCrand = np.load("aaxWCrand_new1.npy")


def output_results():
    plt.plot(aaxSPrand, label='OG on random As')
    plt.plot(aaxWCrand, label='WC on random As')
    plt.legend()
    plt.show()
    print(np.min(aaxSPrand), np.max(aaxSPrand), np.mean(aaxSPrand))
    print(np.min(aaxWCrand), np.max(aaxWCrand), np.mean(aaxWCrand))

Af_new = np.load("Af_new.npy")
Ac_new = np.load("Ac_new.npy")

Af_rob = np.load('working_Af.npy')
Ac_rob = np.load('working_Ac.npy')

