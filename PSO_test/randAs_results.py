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
# aaxSPog = S.area_above_x(per, focusr)

# original model on random As
randConds = np.load("randConds.npy")
randRads = np.load("randRads.npy")
Afs = np.load("afs_rand.npy") # DONT match above vecs
Acs = np.load("acs_rand.npy")

# aaxSPrand = []
# for i in range(len(Afs)):
#     Jrand, _ = S(Af=Afs[i], Ac=Acs[i], per=per, optimizer=1, Jdes=Jdes, 
#     Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)
#     aaxSPrand.append(S.area_above_x(per, focusr, J=Iog))
# aaxSPrand = np.load("og_randAs.npy")
# aaxSPrand = S.area_above_x()
# Afs = None #np.zeros((N, *Af.shape))
# Acs = None #np.zeros((N, *Ac.shape))


def makeOGrand(N=200):
    aaxSPrand = []
    for i in range(N):
        # if i >= 20: exit(0)
        # condvec = np.random.uniform(low=(0.95 * cond_vec), high=(1.05 * cond_vec))
        condvec = randConds[i]
        # rvs = np.array([9.2, 0.5, 0.1])
        # radius_head, thickness_skull, thickness_CSF = np.random.uniform(low=(0.95 * rvs), high=(1.05 * rvs))
        # radvec = np.array([radius_head-thickness_skull-thickness_CSF, radius_head-thickness_skull, radius_head])
        # if (thickness_skull+thickness_CSF) > focus_depth:
        #     i -= 1
        #     continue
        radvec = randRads[i]
        S = SparsePlaceSpherical(cond_vec=condvec, radius_vec=radvec, patch_size=patch_size,
                elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
                custom_grid=custom_grid, theta_elec=None, phi_elec=None)
        v = S.area_above_x(per, focusr, J=Iog)
        print('AREA:', v)
        aaxSPrand.append(v)

        print(str((i+1)/200) + '%', end='\n\n')

    np.save("aaxSPo`g_new1.npy", aaxSPrand)
    return aaxSPrand

aaxSPrand = np.load("aaxSPrand_new1.npy")

# worst case model on original A
WC = SparsePlaceRobust(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
    elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
    custom_grid=custom_grid, theta_elec=None, phi_elec=None)

Iwc, aaxWCog = WC(Af=Af, Ac=Ac, per=per, optimizer=1, Jdes=Jdes, 
    Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)
aaxWCog = S.area_above_x(per, focusr, J=Iwc)

# worst case model on random As
def makeWCrandAs():
    aaxWC = []
    for i in range(len(Afs)):
        WC(Af=Afs[i], Ac=Acs[i], per=per, optimizer=1, Jdes=Jdes, 
        Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)
        aaxWC.append(WC.area_above_x(per, focusr, J=Iwc))
        print(f'\n\n{i/2}% done\n\n')
    return aaxWC
# aaxWCrand = np.asarray(makeWCrandAs())
# np.save("wc_randAs.npy", aaxWCrand)

def makeWCrand(N=200):
    aaxWCrand = []
    for i in range(N):
        # if i >= 20: exit(0)
        # condvec = np.random.uniform(low=(0.95 * cond_vec), high=(1.05 * cond_vec))
        condvec = randConds[i]
        # rvs = np.array([9.2, 0.5, 0.1])
        # radius_head, thickness_skull, thickness_CSF = np.random.uniform(low=(0.95 * rvs), high=(1.05 * rvs))
        # radvec = np.array([radius_head-thickness_skull-thickness_CSF, radius_head-thickness_skull, radius_head])
        # if (thickness_skull+thickness_CSF) > focus_depth:
        #     i -= 1
        #     continue
        radvec = randRads[i]
        S = SparsePlaceSpherical(cond_vec=condvec, radius_vec=radvec, patch_size=patch_size,
                elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
                custom_grid=custom_grid, theta_elec=None, phi_elec=None)
        v = S.area_above_x(per, focusr, J=Iwc)
        print('AREA:', v)
        aaxWCrand.append(v)

        print(str(100*(i+1)/200) + '%', end='\n\n')

    np.save("aaxWCrand_new1.npy", aaxWCrand)
    return aaxWCrand

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

# SnewA = SparsePlaceSpherical(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
#     elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
#     custom_grid=custom_grid, theta_elec=None, phi_elec=None)
# SnewA(Af=Af_new, Ac=Ac_new, per=per, optimizer=1, Jdes=Jdes, 
#     Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)
# AAnewAog = SnewA.area_above_x(per, focusr, J=Iog)
# AAnewAwc = SnewA.area_above_x(per, focusr, J=Iwc)

# WC_moreiter = SparsePlaceRobust(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
#     elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
#     custom_grid=custom_grid, theta_elec=None, phi_elec=None, iters=50)
# Iwc_mi, aaxWCog = WC_moreiter(Af=Af, Ac=Ac, per=per, optimizer=1, Jdes=Jdes, 
#     Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)

# Iwc_mi, Af_mi, Ac_mi = WC_moreiter.SparsePlaceWorstCase(Af=Af, Ac=Ac, getAs=True, Jdes=Jdes, beta=beta, Jsafety=Jsafety)

# SnewA2 = SparsePlaceSpherical(cond_vec=cond_vec, radius_vec=radius_vec, patch_size=patch_size,
#     elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
#     custom_grid=custom_grid, theta_elec=None, phi_elec=None)
# SnewA2(Af=Af_mi, Ac=Ac_mi, per=per, optimizer=1, Jdes=Jdes, 
#     Jsafety=Jsafety, beta=beta, focus_points=focus_points, cancel_points=cancel_points, plot=False)
# AAnewA2og = SnewA.area_above_x(per, focusr, J=Iog)
# AAnewA2wc = SnewA.area_above_x(per, focusr, J=Iwc_mi)

Afrs, Acrs = np.array([]), np.array([])
for i in range(30):
    condvec = randConds[i]
    radvec = randRads[i]
    S2 = SparsePlaceRobust(cond_vec=condvec, radius_vec=radvec, patch_size=patch_size,
            elec_radius=elec_radius, elec_spacing=elec_spacing, max_l=L, spacing=spacing,
            custom_grid=custom_grid, iters=30, theta_elec=None, phi_elec=None)
    Afr, Acr = S.forward_model(focus_points, cancel_points, save=False)
    Afrs = np.append(Afrs, Afr)
    Acrs = np.append(Acrs, Acr)
    np.save("Afrs2.npy", Afrs)
    np.save("Acrs2.npy", Acrs)
    Iwc, Afwc, Acwc = S.SparsePlaceWorstCase(Jdes, Jsafety, Af=Afr, Ac=Acr, beta=beta, getAs=True)
    


    v = S.area_above_x(per, focusr, J=Iog)
    print('AREA:', v)
    aaxSPrand.append(v)