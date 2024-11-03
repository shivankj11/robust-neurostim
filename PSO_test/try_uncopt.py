import numpy as np
import SparsePlace

sigma = np.asarray([1, 1/80, 1])
radii = np.asarray([10-1.1, 10-.6, 10])
cond_vec, radius_vec = sigma, radii
l = np.arange(300) + 1

spherical_model = SparsePlace.SparsePlaceSpherical(cond_vec, radius_vec,
    elec_radius=0.5, elec_spacing=2, patch_size=2, max_l=300, 
    spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None)

focus_points = np.array([(8, np.pi/2, 0)])
cancel_points = ([(4, np.pi/2, 0)])
spherical_model.forward_model(focus_points, cancel_points, 
    print_elec_pattern=False, save=True, save_title_focus="Forward_Focus.npy",
    save_title_cancel="Forward_Cancel.npy")

# try plotting electrodes on head in 3d
