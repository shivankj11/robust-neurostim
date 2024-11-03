import numpy as np
import time
from ElectricGrid import PreDefElecGrid
from TransferFunction import TransferFunction
from CancelPoints import CancelPointsSpherical
import pyshtools as shp_harm
import cvxpy as cvx
import matplotlib.pyplot as plt
import seaborn as sns
import ray

class SparsePlaceSpherical(PreDefElecGrid, TransferFunction, CancelPointsSpherical):

    def __init__(self, cond_vec, radius_vec, elec_radius=0.5, elec_spacing=None, patch_size=None, max_l=300, spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None):

        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten
        self.custom_grid = custom_grid
        if self.custom_grid is False:
            self.patch_size = patch_size * 10**(-2)
            self.elec_spacing = elec_spacing * 10**(-2)
        else:
            self.theta_elec = theta_elec
            self.phi_elec = phi_elec
        self.elec_radius = elec_radius * 10**(-2)
        self.curr_density_calculated = False

    def forward_model(self, focus_points, cancel_points, print_elec_pattern=False, save=True, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy"):

        points = np.vstack((np.array(focus_points), np.array(cancel_points)))
        r = np.unique(points[:,0])

        ## Generate a pseudo-uniform sampling of electrode points
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**(2), patch_size=self.patch_size*10**(2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if print_elec_pattern:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=np.ones(len(elec_lst)))
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()

        ## Calculating the Transfer Function
        self.Af = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac = np.zeros((len(cancel_points), len(elec_lst)))

        previous_len_focus = 0
        previous_len_cancel = 0

        for j in range(len(r)):
            _, tau_Jr, _ = self.calc_tauL(r[j])
            tau_Jr= np.hstack((np.array((0,)), tau_Jr))

            idx_focus = np.abs(focus_points[:, 0] - r[j]) < self.eps
            focus_points_subsampled = focus_points[idx_focus]

            idx_cancel = np.abs(cancel_points[:, 0] - r[j]) < self.eps
            cancel_points_subsampled = cancel_points[idx_cancel]

            ## Calculate the Forward Model
            for i in range(len(elec_lst)):

                start_time = time.time()
                elec_pos = np.vstack((elec_lst[i], ground_elec))
                elec_radii = self.elec_radius*np.ones(len(elec_pos))*10**(2)
                inj_curr = np.array((1,-1))
                self.electrode_sampling(elec_pos, elec_radii, inj_curr)
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)

                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                # elec_grid.plot()
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Multipying the transfer function
                curr_density_coeff = np.zeros(coeff_array.shape)
                curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
                curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
                curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

                curr_density = curr_density_coeff.expand(grid='DH')
                curr_density_data = curr_density.data

                ## Focus Points
                for ii in range(len(focus_points_subsampled)):
                    idx1 = np.searchsorted(self.lat_theta, focus_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, focus_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta) - idx1, idx2]
                    self.Af[ii+j*previous_len_focus,i] = current_at_target

                ## Cancel Points
                for ii in range(len(cancel_points_subsampled)):

                    idx1 = np.searchsorted(self.lat_theta, cancel_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, cancel_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta)-idx1, idx2]
                    self.Ac[ii+j*previous_len_cancel,i]= current_at_target

                print("Electrode Remaining:", len(elec_lst)-i-1, "Time Taken:", time.time()-start_time)

            previous_len_focus = previous_len_focus+len(focus_points_subsampled)
            previous_len_cancel = previous_len_cancel+len(cancel_points_subsampled)
        if save:
            np.save(save_title_focus, self.Af)
            np.save(save_title_cancel, self.Ac)
        return self.Af, self.Ac

    def constrained_Maximization(self, alpha_I, Jsafety, alpha_E, Af=None, Ac=None):

        I = cvx.Variable(Af.shape[1])
        constraints = [cvx.atoms.norm(Ac@I)**2<= alpha_I, cvx.sum(I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(I)<=alpha_E]
        obj = cvx.Maximize(cvx.sum(Af@I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.CVXOPT)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def SparsePlaceVanilla(self, Jdes, Jsafety, Af=None, Ac=None, beta=0.001, J_return=None, area_elec=None):

        if area_elec is None:
            area_elec = np.ones(Af.shape[1])
        I = cvx.Variable(Af.shape[1])

        if J_return is None:
            constraints = [Af @ I == Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        else:
            ones = np.zeros((1, Af.shape[1]))
            ones[len(ones) - 1] = 1
            constraints = [Af @ I == Jdes, ones@I==J_return, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(cvx.multiply(I, area_elec)) <= beta]
        obj = cvx.Minimize(cvx.atoms.norm(Ac@I)) #+beta*cvx.atoms.norm1(I)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS, feastol=1e-4)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def SparsePlaceHinge(self, Jdes, Jsafety, Jtol, Af=None, Ac=None, beta=0.001, J_return=None, area_elec=None):

        if area_elec is None:
            area_elec = np.ones(Af.shape[1])

        I = cvx.Variable(Af.shape[1])
        if J_return is None:
            constraints = [Af @ I == Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        else:
            ones = np.zeros((1,Af.shape[1]))
            ones[len(ones)-1] = 1
            constraints = [Af @ I == Jdes, ones@I==J_return, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(cvx.multiply(I,area_elec))<=beta]
        obj = cvx.Minimize(cvx.sum(cvx.atoms.maximum(0, Ac@I-Jtol*np.ones((Ac.shape[0]))))+cvx.sum(cvx.atoms.maximum(0, -Ac@I-Jtol*np.ones((Ac.shape[0])))))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=1000)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def plot_elec_pttrn(self, J, x_lim=None, y_lim=None, fname=None):
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2), patch_size=self.patch_size * 10 ** (2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2
        J = np.multiply(J,(elec_radius.reshape(J.shape))**2*np.pi)
        self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        grid_electrode.plot(colorbar='right')
        data = self.electrode_spherical_map

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] >= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 100)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 100)

        spacing_x = (x_lim[1] - x_lim[0]) / 100
        spacing_y = (y_lim[1] - y_lim[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")

        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('x-axis (cm)', fontsize='19')
        plt.ylabel('y-axis (cm)', fontsize='19')
        plt.title('Electrode Pattern', fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def calc_voltage(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2),patch_size=self.patch_size * 10 ** (2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        tau_V, _, _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.volatge_calculated = True
        self.voltage_at_target = curr_density.data*10
        return self.voltage_at_target

    def calc_curr_density(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2),patch_size=self.patch_size * 10 ** (2))
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        _, tau_Jr, _ = self.calc_tauL(r)
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_density_calculated = True
        self.curr_density_at_target = curr_density.data
        return self.curr_density_at_target

    def plot_voltage(self, r, J=None, x_limit=None, y_limit=None, fname=None, abs=True):
        if J is None:
           if self.voltage_at_target is False:
                self.calc_voltage(r)
        else:
            self.calc_voltage(r,J)
        return self.plot_given_voltage(r=r, curr_density=self.voltage_at_target, x_limit=x_limit, y_limit=y_limit, fname=fname, abs=abs)

    def plot_given_voltage(self, r, curr_density, x_limit=None, y_limit=None, abs=True, fname=None):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = curr_density.flatten()
        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]
        max_v = np.max(np.abs(data_flatten))
        data_flatten = data_flatten[idx]




        if x_limit is None:
            x_limit = np.array((np.min(x), np.max(x)))
        if y_limit is None:
            y_limit = np.array((np.min(y), np.max(y)))

        x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 100)
        y_discretize = np.arange(y_limit[0], y_limit[1] + self.eps, (y_limit[1] - y_limit[0]) / 100)

        spacing_x = (x_limit[1] - x_limit[0]) / 100
        spacing_y = (y_limit[1] - y_limit[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))

        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])

        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        if abs is True:
            sns.heatmap(np.abs(data_projected), cmap="jet")
        else:
            sns.heatmap(data_projected, cmap="jet")


        labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        plt.xticks(x_ticks, labels_x, fontsize='13')
        plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xlabel('x-axis (mm)', fontsize='17')
        plt.ylabel('y-axis (mm)', fontsize='17')
        plt.title('Voltage ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        if fname is not None:
            plt.savefig(fname)
        plt.tight_layout()
        plt.show()
        print("Position of maximum [%s,%s]"%(max_x,max_y))
        return max_v

    def plot_given_voltage_1d(self, r, curr_density, length=1, angle=0, normalize=False, fname=None, num_points=500, show=True):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = curr_density.flatten()
        data_flatten = data_flatten[idx]
        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]

        lin_coord_x, lin_coord_y = np.arange(0, length+self.eps, length/num_points)*np.cos(angle), np.arange(0, length+self.eps, length/num_points)*np.sin(angle)
        lin_coord_x, lin_coord_y = max_x+lin_coord_x, max_y+lin_coord_y
        data_projected = np.zeros((len(lin_coord_x),))
        spacing = length / num_points


        for i in range(len(lin_coord_y)):
            data_projected[i] = np.mean(data_flatten[np.where(np.square(x - lin_coord_x[i]) + np.square(y - lin_coord_y[i]) <= spacing**2)])

        if normalize:
            data_projected = data_projected/np.max(np.abs(data_projected))

        plt.plot(np.arange(0, length+self.eps, length/num_points)*10, data_projected)

        # labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        # labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        # labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        # labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        # labels_y = np.flip(np.array(labels_y))
        # x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        # y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        # plt.xticks(x_ticks, labels_x, fontsize='13')
        # plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xticks(fontsize='13')
        plt.yticks(fontsize='13')
        plt.xlabel('Distance from Centre (mm)', fontsize='17')
        plt.ylabel('Field Voltage (V)', fontsize='17')
        plt.title('Current Density ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
        return [max_x, max_y], data_projected

    def plot_voltage_1d(self, r, J=None, length=1, angle=0, fname=None, normalize=True, num_points=500, show=True):
        if J is None:
           if self.voltage_at_target is False:
                self.calc_voltage(r)
        else:
            self.calc_voltage(r,J)
        return self.plot_given_voltage_1d(r=r, curr_density=self.voltage_at_target, length=length, angle=angle, fname=fname, normalize=normalize, num_points=num_points, show=show)

    def plot_curr_density(self, r, J=None, x_limit=None, y_limit=None, fname=None, abs=True):
        if J is None:
           if self.curr_density_calculated is False:
                self.calc_curr_density(r)
        else:
            self.calc_curr_density(r,J)
        return self.plot_given_curr_density(r=r, curr_density=self.curr_density_at_target, x_limit=x_limit, y_limit=y_limit, fname=fname, abs=abs)

    def plot_curr_density_1d(self, r, J=None, length=1, angle=0, fname=None, normalize=True, num_points=500, show=True):
        if J is None:
           if self.curr_density_calculated is False:
                self.calc_curr_density(r)
        else:
            self.calc_curr_density(r,J)
        return self.plot_given_curr_density_1d(r=r, curr_density=self.curr_density_at_target, length=length, angle=angle, fname=fname, normalize=normalize, num_points=num_points, show=show)

    def get_max_location(self,  r, J=None):
        if J is None:
           if self.curr_density_calculated is False:
                self.calc_curr_density(r)
        else:
            self.calc_curr_density(r,J)
        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(
            spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = self.curr_density_at_target.flatten()
        data_flatten = data_flatten[idx]

        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]
        return [max_x, max_y]

    def plot_given_curr_density(self, r, curr_density, x_limit=None, y_limit=None, abs=True, fname=None):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = curr_density.flatten()
        data_flatten = data_flatten[idx]

        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]


        if x_limit is None:
            x_limit = np.array((np.min(x), np.max(x)))
        if y_limit is None:
            y_limit = np.array((np.min(y), np.max(y)))

        x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 200)
        y_discretize = np.arange(y_limit[0], y_limit[1] + self.eps, (y_limit[1] - y_limit[0]) / 200)

        spacing_x = (x_limit[1] - x_limit[0]) / 200
        spacing_y = (y_limit[1] - y_limit[0]) / 200
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))

        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])

        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        if abs is True:
            sns.heatmap(np.abs(data_projected), cmap="jet")
        else:
            sns.heatmap(data_projected, cmap="jet")


        labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        plt.xticks(x_ticks, labels_x, fontsize='13')
        plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xlabel('x-axis (mm)', fontsize='17')
        plt.ylabel('y-axis (mm)', fontsize='17')
        plt.title('Current Density ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)

        plt.show()
        return [max_x, max_y]

    def plot_given_curr_density_1d(self, r, curr_density, length=1, angle=0, normalize=False, fname=None, num_points=500, show=True):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = curr_density.flatten()
        data_flatten = data_flatten[idx]
        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]

        lin_coord_x, lin_coord_y = np.arange(0, length+self.eps, length/num_points)*np.cos(angle), np.arange(0, length+self.eps, length/num_points)*np.sin(angle)
        lin_coord_x, lin_coord_y = max_x+lin_coord_x, max_y+lin_coord_y
        data_projected = np.zeros((len(lin_coord_x),))
        spacing = length / num_points


        for i in range(len(lin_coord_y)):
            data_projected[i] = np.mean(data_flatten[np.where(np.square(x - lin_coord_x[i]) + np.square(y - lin_coord_y[i]) <= spacing**2)])

        if normalize:
            data_projected = data_projected/np.max(np.abs(data_projected))

        plt.plot(np.arange(0, length+self.eps, length/num_points)*10, data_projected)

        # labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        # labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        # labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        # labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        # labels_y = np.flip(np.array(labels_y))
        # x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        # y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        # plt.xticks(x_ticks, labels_x, fontsize='13')
        # plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xticks(fontsize='13')
        plt.yticks(fontsize='13')
        plt.xlabel('Distance from Centre (mm)', fontsize='17')
        plt.ylabel('Field Amplitude (mA/cm$^2$)', fontsize='17')
        plt.title('Current Density ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
        return [max_x, max_y], data_projected

    # def plot_currdensity_rtheta(self,  J=None, theta_limit=None, r_limit=None, fname=None, abs=True):
    #
    #     r = np.arange(r_limit[0], r_limit[1]+self.eps, (r_limit[1]-r_limit[0])/100)
    #     theta = np.arange(theta_limit[0], theta_limit[1]+self.eps, (theta_limit[1]-theta_limit[0]))
    #     if x_limit is None:
    #         x_limit = np.array((np.min(x), np.max(x)))
    #
    #     x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 100)
    #     for i in range(len(r)):
    #
    #         curr_density = self.calc_curr_density(r=r[i], J=J)
    #         long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
    #         long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
    #         spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))
    #
    #         idx = np.abs(spherical_map_flatten[:,0])<self.eps
    #         spherical_map_flatten = spherical_map_flatten[idx]
    #         data_flatten = curr_density.flatten()
    #         data_flatten = data_flatten[idx]
    #
    #         ### Plotting the first half
    #         idx = spherical_map_flatten[:, 1] <= 0
    #
    #         x, z = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r*np.sin(spherical_map_flatten[idx,1])
    #         data_flatten = curr_density.flatten()
    #         data_flatten = data_flatten[idx]
    #
    #
    #         max_idx = np.argmax(np.abs(data_flatten))
    #         max_x = x[max_idx],
    #
    #
    #
    #         spacing_x = (x_limit[1] - x_limit[0]) / 100
    #
    #
    #         for i in range(len(x_discretize)):
    #             for j in range(len(y_discretize)):
    #                 data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(
    #                     y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
    #
    #     data_projected = np.transpose(data_projected)
    #     data_projected = np.flip(data_projected, axis=0)
    #     if abs is True:
    #         sns.heatmap(np.abs(data_projected), cmap="jet")
    #     else:
    #         sns.heatmap(data_projected, cmap="jet", vmax=0.1281623990903234, vmin=0.0011353618018483764)
    #
    #     labels_x = np.linspace(x_limit[0], x_limit[1], 11)
    #     labels_x = [str(round(labels_x[i], 1))[0:4] for i in range(len(labels_x))]
    #     labels_y = np.linspace(y_limit[0], y_limit[1], 11)
    #     labels_y = [str(round(labels_y[i], 1))[0:4] for i in range(len(labels_y))]
    #     labels_y = np.flip(np.array(labels_y))
    #     x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
    #     y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
    #
    #     plt.xticks(x_ticks, labels_x, fontsize='15')
    #     plt.yticks(y_ticks, labels_y, fontsize='15')
    #     plt.xlabel('x-axis (cm)', fontsize='19')
    #     plt.ylabel('y-axis (cm)', fontsize='19')
    #     plt.title('Current Density (' + str(round(self.r_max * 10 ** 2 - r, 2)) + " cm)", fontsize='21')
    #     plt.savefig(fname)
    #     plt.tight_layout()
    #     plt.show()

    def area_above_x(self, per, r, plot=False, J=None, x_limit=None, y_limit=None, fname=None, density_thresh=None, print_area=True):
        if self.curr_density_calculated is False or J is not None:
            self.calc_curr_density(r=r, J=J)
        area = self.area_above_x_curr_density(per=per, r=r, plot=plot, curr_density=self.curr_density_at_target, x_limit=x_limit, y_limit=y_limit, fname=fname, density_thresh=density_thresh, print_area=print_area)
        return area

    def get_area_above_x_density(self, per, r, J=None, x_limit=None, y_limit=None, density_thresh=None):
        if self.curr_density_calculated is False or J is not None:
            self.calc_curr_density(r=r, J=J)

        max_density = np.max(self.curr_density_at_target)
        if density_thresh is None:
            density_thresh = max_density * per
        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))
        idx = spherical_map_flatten[:, 1] <= 0
        spherical_map_flatten = spherical_map_flatten[idx]
        data_flatten = self.curr_density_at_target.flatten()
        data_flatten = data_flatten[idx]
        points_above_thresh = spherical_map_flatten[data_flatten > density_thresh]
        del_theta = np.abs(self.lat_theta[0] - self.lat_theta[1])
        del_phi = np.abs(self.long_phi[0] - self.long_phi[1])
        area = r ** 2 * del_theta * del_phi * np.sum(np.cos(points_above_thresh[:, 1]))
        #print("Area above " + str(int(per * 100)) + " of the max:", area, "cm^2")
        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        # data_flatten = curr_density.flatten()
        idx_1 = data_flatten > density_thresh
        idx_2 = data_flatten <= density_thresh
        data_flatten[idx_1] = 1
        data_flatten[idx_2] = 0
        data_flatten = data_flatten[idx]
        if x_limit is None:
            x_limit = np.array((np.min(x), np.max(x)))
        if y_limit is None:
            y_limit = np.array((np.min(y), np.max(y)))

        x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 100)
        y_discretize = np.arange(y_limit[0], y_limit[1] + self.eps, (y_limit[1] - y_limit[0]) / 100)

        spacing_x = (x_limit[1] - x_limit[0]) / 100
        spacing_y = (y_limit[1] - y_limit[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        data_projected = data_projected > 0.5
        return data_projected

    def area_above_x_curr_density(self, per, r, curr_density, plot=False, print_area=True, x_limit=None, y_limit=None, fname=None, density_thresh=None):

        max_density = np.max(curr_density)
        if density_thresh is None:
            density_thresh = max_density * per
        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))
        idx = spherical_map_flatten[:, 1] <= 0
        spherical_map_flatten = spherical_map_flatten[idx]
        data_flatten = curr_density.flatten()
        data_flatten = data_flatten[idx]
        points_above_thresh = spherical_map_flatten[data_flatten > density_thresh]
        del_theta = np.abs(self.lat_theta[0] - self.lat_theta[1])
        del_phi = np.abs(self.long_phi[0] - self.long_phi[1])
        area = r**2 * del_theta * del_phi * np.sum(np.cos(points_above_thresh[:, 1]))
        if print_area:
            print("Area above " + str(int(per * 100)) + " of the max:", area, "cm^2")
        if plot is True:
            idx = spherical_map_flatten[:, 1] <= 0
            x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
            # data_flatten = curr_density.flatten()
            idx_1 = data_flatten > density_thresh
            idx_2 = data_flatten <= density_thresh
            data_flatten[idx_1] = 1
            data_flatten[idx_2] = 0
            data_flatten = data_flatten[idx]

            if x_limit is None:
                x_limit = np.array((np.min(x), np.max(x)))
            if y_limit is None:
                y_limit = np.array((np.min(y), np.max(y)))

            x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 100)
            y_discretize = np.arange(y_limit[0], y_limit[1] + self.eps, (y_limit[1] - y_limit[0]) / 100)

            spacing_x = (x_limit[1] - x_limit[0]) / 100
            spacing_y = (y_limit[1] - y_limit[0]) / 100
            data_projected = np.zeros((len(x_discretize), len(y_discretize)))
            for i in range(len(x_discretize)):
                for j in range(len(y_discretize)):
                    data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
            data_projected = np.transpose(data_projected)
            data_projected = np.flip(data_projected, axis=0)
            data_projected = data_projected>0.5
            sns.heatmap(np.abs(data_projected), cmap="jet")
            labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
            labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
            labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
            labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
            labels_y = np.flip(np.array(labels_y))
            x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
            y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
            plt.xticks(x_ticks, labels_x, fontsize='13')
            plt.yticks(y_ticks, labels_y, fontsize='13')
            plt.xlabel('x-axis (mm)', fontsize='17')
            plt.ylabel('y-axis (mm)', fontsize='17')
            plt.title('Region Activated (' + str(round(self.r_max * 10 ** (2) - r,2)) +' cm)', fontsize='21')
            plt.tight_layout()
            plt.savefig(fname)
            plt.show()
        return area

    def volume_above_x(self, per, r, dr=0.005, plot=False, J=None, density_thresh=None):
        if J is -1:
            return -1
        vol = 0
        up_flag = True
        init = True
        self.calc_curr_density(r=r, J=J)
        if density_thresh is None:
            max_density = np.max(np.abs(self.curr_density_at_target))
            density_thresh = max_density * per
        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))
        idx = spherical_map_flatten[:, 1] <= 0
        spherical_map_flatten = spherical_map_flatten[idx]
        data_flatten = self.curr_density_at_target.flatten()
        data_flatten = data_flatten[idx]
        points_above_thresh = spherical_map_flatten[data_flatten > density_thresh]
        del_theta = np.abs(self.lat_theta[0] - self.lat_theta[1])
        del_phi = np.abs(self.long_phi[0] - self.long_phi[1])
        vol = vol + dr * r ** 2 * del_theta * del_phi * np.sum(np.cos(points_above_thresh[:, 1]))
        r = r - dr
        while up_flag:

            self.calc_curr_density(r=r, J=J)
            long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
            long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
            spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))
            idx = spherical_map_flatten[:, 1] <= 0
            spherical_map_flatten = spherical_map_flatten[idx]
            data_flatten = self.curr_density_at_target.flatten()
            data_flatten = data_flatten[idx]
            points_above_thresh = spherical_map_flatten[data_flatten>density_thresh]
            if len(points_above_thresh) == 0:
                up_flag=False
                break
            del_theta = np.abs(self.lat_theta[0]-self.lat_theta[1])
            del_phi = np.abs(self.long_phi[0]-self.long_phi[1])
            vol = vol+ dr*r**2*del_theta*del_phi*np.sum(np.cos(points_above_thresh[:,1]))

            r=r-dr
        return vol, r

    def run_SparsePlaceVanilla(self, per, Jdes, Jsafety, focus_points, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):
        print(">>>>>>>>>Running Vanilla Loss SparsePlace<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_sparseplace_vanilla = self.SparsePlaceVanilla(Jdes=Jdes, Jsafety=Jsafety, Af=self.Af, Ac=self.Ac, beta=beta)
        else:
            self.J_sparseplace_vanilla = self.SparsePlaceVanilla(Jdes=Jdes, Jsafety=Jsafety, Af=Af, Ac=Ac, beta=beta)

        if self.J_sparseplace_vanilla is None:
            return -1, -1
        else:
            curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_vanilla)
            print("Error Factor:", Jdes / np.max(curr_density))
            self.J_sparseplace_vanilla = Jdes / np.max(curr_density) * self.J_sparseplace_vanilla
            if print_elec_pattern:
                self.plot_elec_pttrn(self.J_sparseplace_vanilla, fname=fname_elec, x_lim=np.array((-0.2,0.2)), y_lim=np.array((-0.2,0.2)))
            if plot is True:
                max_x, max_y = self.plot_curr_density(r=np.min(focus_points[:,0]),  J=self.J_sparseplace_vanilla, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_density, abs=False)
                print("Coordinates of the centre:", max_x, max_y, 'for Jdes:', Jdes)
            if density_thresh is None:
                area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_vanilla, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_area)
            else:
                area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_vanilla, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), density_thresh=density_thresh, fname=fname_area)

        return self.J_sparseplace_vanilla, area

    def run_constrained_Maximiztion(self, per, alpha_I, Jsafety, focus_points,alpha_E=1e9, cancel_points=None, Af=None, Ac=None, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, density_thresh=None):
        print(">>>>>>>>>Running Constrained Maximization<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_const_max = self.constrained_Maximization(alpha_I=alpha_I, alpha_E=alpha_E, Jsafety=Jsafety, Af=self.Af, Ac=self.Ac)
        else:
            self.J_const_max = self.constrained_Maximization(alpha_I=alpha_I, alpha_E=alpha_E, Jsafety=Jsafety, Af=Af, Ac=Ac)
        self.plot_elec_pttrn(self.J_const_max, x_lim=np.array((-0.2,0.2)), y_lim=np.array((-0.2,0.2)))
        # if plot is True:
        self.plot_curr_density(r=np.min(focus_points[:,0]),  J=self.J_const_max)
        if density_thresh is None:
            area = self.area_above_x(r=np.min(focus_points[:,0]), per=per, plot=True, J=self.J_const_max, x_limit=np.array((-0.2,0.2)), y_limit=np.array((-0.2,0.2)), abs=False)
        else:
            area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=True, J=self.J_const_max, x_limit=np.array((-0.2, 0.2)), y_limit=np.array((-0.2, 0.2)), density_thresh=density_thresh)
        return self.J_const_max, area

    def run_SparsePlaceHinge(self, per, Jdes, Jsafety, focus_points, Jtol, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):

        print(">>>>>>>>>Running Hinge Loss SparsePlace<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_sparseplace_hinge = self.SparsePlaceHinge(Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol, Af=self.Af, Ac=self.Ac, beta=beta)
        else:
            self.J_sparseplace_hinge = self.SparsePlaceHinge(Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol, Af=Af, Ac=Ac, beta=beta)

        if self.J_sparseplace_hinge is None:
            return -1,-1
        else:
            curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_hinge)
            print("Error Factor:", Jdes / np.max(curr_density))
            self.J_sparseplace_hinge = Jdes / np.max(curr_density) * self.J_sparseplace_hinge
            if print_elec_pattern:
                self.plot_elec_pttrn(self.J_sparseplace_hinge, fname=fname_elec, x_lim=np.array((-0.2, 0.2)), y_lim=np.array((-0.2, 0.2)))
            if plot is True:
                max_x, max_y = self.plot_curr_density(r=np.min(focus_points[:,0]), J=self.J_sparseplace_hinge, fname=fname_density, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), abs=False)
                print("Coordinates of the centre:", max_x, max_y, 'for Jdes:', Jdes)
            if density_thresh is None:
                area = self.area_above_x(r=np.min(focus_points[:,0]), per=per, plot=plot, J=self.J_sparseplace_hinge,x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_area)
            else:
                area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_hinge, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_area, density_thresh=density_thresh)

        return self.J_sparseplace_hinge, area

    def __call__(self, Jdes, Jsafety, focus_points, per=0.8,optimizer =1, Jtol=None, alpha_E=1e9, alpha_I=None, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):

        if optimizer == 1:
            J_elec, area = self.run_SparsePlaceVanilla(per=per,Jdes=Jdes, Jsafety=Jsafety, focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        elif optimizer == 2:
            J_elec, area = self.run_SparsePlaceHinge(per=per,Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol,focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        elif optimizer == 3:
            J_elec, area = self.run_constrained_Maximiztion(per=per,alpha_E=alpha_E, Jsafety=Jsafety, alpha_I=alpha_I,focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, density_thresh=density_thresh)

        return J_elec, area

class SparsePlaceConc(SparsePlaceSpherical):

    def __init__(self, cond_vec, radius_vec, width, num_rings, ring_spacing, start_radius, center, max_l=300, spacing=0.1):

        self.eps = 0.0001 * 10 ** (-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing = spacing * 10 ** (-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten

        self.num_rings = num_rings
        self.center = center
        self.width = width*10**(-2)
        self.ring_spacing = ring_spacing*10**(-2)
        self.start_radius = start_radius*10**(-2)
        self.curr_density_calculated = False

    def forward_model(self, focus_points, cancel_points, print_elec_pattern=True, save=True, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy"):
        points = np.vstack((np.array(focus_points), np.array(cancel_points)))
        r = np.unique(points[:,0])
        ## Generate a pseudo-uniform sampling of electrode points
        radii, width_arr, center, ground_elec = self.conc_ring_patch(spacing=self.ring_spacing*10**(2), width=self.width*10**(2), num_rings=self.num_rings, start_radius=self.start_radius*10**(2), center=self.center)
        if print_elec_pattern:
            self.sample_conc_ring(center=center*np.ones(((len(radii),2))), radius=radii*10**(2), width=width_arr*10**(2), injected_curr=np.ones(self.num_rings))
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()

        ## Calculating the Transfer Function
        self.Af = np.zeros((len(focus_points), self.num_rings))
        self.Ac = np.zeros((len(cancel_points), self.num_rings))

        previous_len_focus = 0
        previous_len_cancel = 0

        radii_ground = np.min(radii)
        center_ground = np.array((-np.pi/2.0,0))
        width_ground = np.min(width_arr)
        area_ground = np.pi * (radii_ground ** 2 - (radii_ground - width_ground) ** 2)
        area_ground = area_ground * 1e4
        for j in range(len(r)):
            _, tau_Jr, _ = self.calc_tauL(r[j])
            tau_Jr= np.hstack((np.array((0,)), tau_Jr))

            idx_focus = np.abs(focus_points[:, 0] - r[j]) < self.eps
            focus_points_subsampled = focus_points[idx_focus]

            idx_cancel = np.abs(cancel_points[:, 0] - r[j]) < self.eps
            cancel_points_subsampled = cancel_points[idx_cancel]

            ## Calculate the Forward Model
            for i in range(self.num_rings):

                start_time = time.time()

                radius_el = np.array((radii[i], radii_ground))
                width_el = np.array((width_arr[i], width_ground))
                center_el = np.vstack((center.reshape(-1,2), center_ground.reshape((-1,2))))
                area_el = np.pi*(radii[i]**2-(radii[i]-width_arr[i])**2)
                area_el = area_el*1e4
                inj_curr = np.array((1/area_el,-1/area_ground))

                self.sample_conc_ring(center=center_el, radius=radius_el*10**(2), width=width_el*10**(2), injected_curr=inj_curr)
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)

                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                # elec_grid.plot()
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Multipying the transfer function
                curr_density_coeff = np.zeros(coeff_array.shape)
                curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
                curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
                curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

                curr_density = curr_density_coeff.expand(grid='DH')
                curr_density_data = curr_density.data

                ## Focus Points
                for ii in range(len(focus_points_subsampled)):
                    idx1 = np.searchsorted(self.lat_theta, focus_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, focus_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta) - idx1, idx2]
                    self.Af[ii+j*previous_len_focus,i] = current_at_target

                ## Cancel Points
                for ii in range(len(cancel_points_subsampled)):

                    idx1 = np.searchsorted(self.lat_theta, cancel_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, cancel_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta)-idx1, idx2]
                    self.Ac[ii+j*previous_len_cancel,i]= current_at_target

                print("Electrode Remaining:", self.num_rings-i-1, "Time Taken:", time.time()-start_time)

            previous_len_focus = previous_len_focus+len(focus_points_subsampled)
            previous_len_cancel = previous_len_cancel+len(cancel_points_subsampled)
        if save:
            np.save(save_title_focus, self.Af)
            np.save(save_title_cancel, self.Ac)
        return self.Af, self.Ac

    def calc_curr_density(self, r, J=None):
        ## Electrode Density
        radii, width_arr, center, ground_elec = self.conc_ring_patch(spacing=self.ring_spacing * 10 ** (2), width=self.width * 10 ** (2), num_rings=self.num_rings, start_radius=self.start_radius * 10 ** (2), center=self.center)
        if J is None:
            J = self.J_sparseplace_vanilla

        J_density = np.zeros(J.shape)
        for i in range(len(radii)):
            area = np.pi*(radii[i]**2-(radii[i]-width_arr[i])**2)*1e4
            J_density[i] = J[i]/area

        self.sample_conc_ring(center=center*np.ones((len(radii),2)), radius=radii*10**(2), width=width_arr*10**(2), injected_curr=J_density)

        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        _, tau_Jr, _ = self.calc_tauL(r)
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_density_calculated = True
        self.curr_density_at_target = curr_density.data
        return self.curr_density_at_target

    def plot_elec_pttrn(self, J=None, x_start=-1.2, x_end=1.2, y_start=-1.2, y_end=1.2):
        radii, width_arr, center, ground_elec = self.conc_ring_patch(spacing=self.ring_spacing * 10 ** (2), width=self.width * 10 ** (2), num_rings=self.num_rings, start_radius=self.start_radius * 10 ** (2), center=self.center)
        radii = radii*10**2
        center_x, center_y = self.r_max*10**(2)*np.cos(center[0])*np.cos(center[1]),self.r_max*10**(2)*np.cos(center[0])*np.sin(center[1])
        dtheta = np.arange(0,2*np.pi+2*np.pi/100, 2*np.pi/100)
        for i in range(len(radii)):
            if np.abs(J[i])>0.01*np.max(np.abs(J)):
                elec_loc_x, elec_loc_y = center_x+radii[i]*np.cos(dtheta), center_y+radii[i]*np.sin(dtheta)
                if J[i]>0:
                    plt.plot(elec_loc_x, elec_loc_y, color='r')
                else:
                    plt.plot(elec_loc_x, elec_loc_y, color='b')

                x, y = radii[i]*np.cos(i*np.pi/2), radii[i]*np.sin(i*np.pi/2)
                if J[i]>0:
                    plt.annotate(str(J[i])[0:3], (x, y), arrowprops = dict(facecolor='black', width=3, headwidth=0.2, headlength=0.4), fontsize=12)
                else:
                    plt.annotate(str(J[i])[0:4], (x, y),arrowprops=dict(facecolor='black', width=3, headwidth=0.2, headlength=0.4),fontsize=12)


        plt.xlabel("X-direction in (cm)", fontsize='16')
        plt.ylabel("Y-direction in (cm)", fontsize='16')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Electrode Pattern", fontsize='18')
        plt.tight_layout()

        plt.savefig("Figures/EMBCResults/Elec_Pattern_hinge.png")
        plt.show()

@ray.remote
class SparsePlaceRectangular(SparsePlaceSpherical):

    def __init__(self, cond_vec, radius_vec, num_elec, elec_radius, elec_spacing, max_l=300, spacing=0.1):

        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten
        self.num_elec = num_elec
        self.elec_spacing = elec_spacing * 10**(-2)

        self.elec_radius = elec_radius * 10**(-2)
        self.curr_density_calculated = False
        self.curr_norm_calculated = False

    def forward_model(self, focus_points, cancel_points, print_elec_pattern=False, save=True, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy"):

        points = np.vstack((np.array(focus_points), np.array(cancel_points)))
        r = np.unique(points[:,0])

        ## Generate a pseudo-uniform sampling of electrode points

        elec_lst, ground_elec, elec_radius = self.approx_rectangular(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec)

        if print_elec_pattern:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=np.ones(len(elec_lst)))
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()

        ## Calculating the Transfer Function
        self.Af = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac = np.zeros((len(cancel_points), len(elec_lst)))

        self.Af_theta = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac_theta = np.zeros((len(cancel_points), len(elec_lst)))

        self.Af_phi = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac_phi = np.zeros((len(cancel_points), len(elec_lst)))

        previous_len_focus = 0
        previous_len_cancel = 0

        for j in range(len(r)):
            tau_V, tau_Jr, _ = self.calc_tauL(r[j])
            tau_Jr= np.hstack((np.array((0,)), tau_Jr))
            tau_V = np.hstack((np.array((0,)), tau_V))

            idx_focus = np.abs(focus_points[:, 0] - r[j]) < self.eps
            focus_points_subsampled = focus_points[idx_focus]

            idx_cancel = np.abs(cancel_points[:, 0] - r[j]) < self.eps
            cancel_points_subsampled = cancel_points[idx_cancel]

            ## Calculate the Forward Model
            for i in range(len(elec_lst)):

                start_time = time.time()
                elec_pos = np.vstack((elec_lst[i], ground_elec))
                elec_radii = self.elec_radius*np.ones(len(elec_pos))*10**(2)
                inj_curr = np.array((1,-1))
                self.electrode_sampling(elec_pos, elec_radii, inj_curr)
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)

                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                # elec_grid.plot()
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Multipying the transfer function Current Density in r direction
                curr_density_coeff = np.zeros(coeff_array.shape)
                curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
                curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
                curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)
                curr_density = curr_density_coeff.expand(grid='DH')
                curr_density_data = curr_density.data ## mA/cm^2

                ## Calculating Voltage

                voltage_coeff = np.zeros(coeff_array.shape)
                voltage_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
                voltage_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
                voltage_coeff = shp_harm.SHCoeffs.from_array(voltage_coeff)

                voltage = voltage_coeff.expand(grid='DH')
                voltage_data = voltage.data * 10 ## V

                ## Calculate current density in theta direction
                del_theta = np.abs(self.lat_theta[0] - self.lat_theta[1])
                curr_density_theta = np.zeros(voltage_data.shape)
                curr_density_theta[0] = (voltage_data[1]-voltage_data[0])/(r[j]*10**(-2)* del_theta) ## V/m
                curr_density_theta[curr_density_theta.shape[0]-1] = (voltage_data[curr_density_theta.shape[0]-1]-voltage_data[curr_density_theta.shape[0]-2])/(r[j]*10**(-2)*del_theta)
                for ii in range(voltage_data.shape[0]-2):
                    curr_density_theta[ii+1] = (voltage_data[ii+2]-voltage_data[ii])/(2*r[j]*10**(-2)*del_theta)
                curr_density_theta = self.cond_vec[len(self.cond_vec)-1]*curr_density_theta/10 # mA/cm^2, assuming that r lies in brain

                ## Calculate current density in phi direction
                del_phi = np.abs(self.long_phi[0] - self.long_phi[1])
                curr_density_phi = np.zeros(voltage_data.shape)
                cos_theta = np.cos(self.lat_theta)
                cos_theta[0] = 1
                cos_theta[len(cos_theta) - 1] = 1
                curr_density_phi[:,0] = (voltage_data[:,1]-voltage_data[:,0])/(r[j]*10**(-2)*del_phi*cos_theta)
                curr_density_phi[:, curr_density_phi.shape[1]-1] = (voltage_data[:,curr_density_phi.shape[1]-1]-voltage_data[:,curr_density_phi.shape[1]-2]) / (r[j]*10**(-2)*del_phi*cos_theta)
                for ii in range(voltage_data.shape[1]-2):
                    curr_density_phi[:,ii+1] = (voltage_data[:,ii+2]-voltage_data[:,ii])/(2*r[j]*10**(-2)*del_phi*cos_theta)
                curr_density_phi = self.cond_vec[len(self.cond_vec) - 1]*curr_density_phi / 10

                ## Focus Points
                for ii in range(len(focus_points_subsampled)):
                    idx1 = np.searchsorted(self.lat_theta, focus_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, focus_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta) - idx1, idx2]
                    current_at_target_theta = curr_density_theta[len(self.lat_theta) - idx1, idx2]
                    current_at_target_phi = curr_density_phi[len(self.lat_theta) - idx1, idx2]

                    self.Af[ii+j*previous_len_focus,i] = current_at_target
                    self.Af_theta[ii+j*previous_len_focus,i] = current_at_target_theta
                    self.Af_phi[ii+j*previous_len_focus,i] = current_at_target_phi

                ## Cancel Points
                for ii in range(len(cancel_points_subsampled)):

                    idx1 = np.searchsorted(self.lat_theta, cancel_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, cancel_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta)-idx1, idx2]
                    current_at_target_theta = curr_density_theta[len(self.lat_theta)-idx1, idx2]
                    current_at_target_phi = curr_density_phi[len(self.lat_theta)-idx1, idx2]

                    self.Ac[ii+j*previous_len_focus, i] = current_at_target
                    self.Ac_theta[ii+j*previous_len_focus, i] = current_at_target_theta
                    self.Ac_phi[ii+j*previous_len_focus, i] = current_at_target_phi

                print("Electrode Remaining:", len(elec_lst)-i-1, "Time Taken:", time.time()-start_time)

            previous_len_focus = previous_len_focus+len(focus_points_subsampled)
            previous_len_cancel = previous_len_cancel+len(cancel_points_subsampled)
        if save:
            np.save(save_title_focus+"_r.npy", self.Af)
            np.save(save_title_cancel+"_r.npy", self.Ac)
            np.save(save_title_focus + "_theta.npy", self.Af_theta)
            np.save(save_title_cancel + "_theta.npy", self.Ac_theta)
            np.save(save_title_focus + "_phi.npy", self.Af_phi)
            np.save(save_title_cancel + "_phi.npy", self.Ac_phi)
        return self.Af, self.Ac

    def get_Af_and_Ac(self):
        return [self.Af, self.Af_theta, self.Af_phi], [self.Ac, self.Ac_theta, self.Ac_phi]

    def calc_curr_density(self, r, J=None):
        ## Electrode Density
        elec_lst, ground_elec, elec_radius = self.approx_rectangular(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec)

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        _, tau_Jr, _ = self.calc_tauL(r)
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_density_calculated = True
        self.curr_density_at_target = curr_density.data
        return self.curr_density_at_target

    def calc_voltage(self, r, J=None):
        ## Electrode Density
        elec_lst, ground_elec, elec_radius = self.approx_rectangular(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec)

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)

        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        tau_V, _, _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        voltage = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.volatge_calculated = True
        self.voltage_at_target = voltage.data * 10
        return self.voltage_at_target

    def calc_Jnorm(self, r, J=None):
        ## Electrode Density
        elec_lst, ground_elec, elec_radius = self.approx_rectangular(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec)

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)

        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        tau_V,tau_Jr , _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        voltage = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.volatge_calculated = True
        self.voltage_at_target = voltage.data*10
        latitude = voltage.lats()
        latitude = latitude/360*2*np.pi

        del_theta = np.abs(self.lat_theta[0]-self.lat_theta[1])
        self.curr_density_theta = np.zeros(self.voltage_at_target.shape)
        self.curr_density_theta[0] = (self.voltage_at_target[1]-self.voltage_at_target[0])/(r*10**(-2)*del_theta)
        self.curr_density_theta[self.curr_density_theta.shape[0]-1] = (self.voltage_at_target[self.curr_density_theta.shape[0]-1] - self.voltage_at_target[self.curr_density_theta.shape[0]-2]) / (r * 10 ** (-2) * del_theta)
        for i in range(self.voltage_at_target.shape[0]-2):
            self.curr_density_theta[i+1] = (-self.voltage_at_target[i]+self.voltage_at_target[i+2])/(2*r*10**(-2)*del_theta)
        self.curr_density_theta = self.cond_vec[len(self.cond_vec)-1]*self.curr_density_theta/10

        del_phi = np.abs(self.long_phi[0] - self.long_phi[1])
        self.curr_density_phi = np.zeros(self.voltage_at_target.shape)
        cos_theta = np.cos(latitude)
        cos_theta[0] = 1
        cos_theta[len(cos_theta)-1]=1
        self.curr_density_phi[:,0] = (self.voltage_at_target[:,1] - self.voltage_at_target[:,0])/(r*10**(-2)*del_phi*cos_theta)
        self.curr_density_phi[:,self.curr_density_theta.shape[1]-1] = (self.voltage_at_target[:,self.curr_density_theta.shape[1]-1] -self.voltage_at_target[:,self.curr_density_theta.shape[1]-2]) / (r*10**(-2)*del_phi*cos_theta)
        for i in range(self.voltage_at_target.shape[1] - 2):
            self.curr_density_phi[:, i+1] = (-self.voltage_at_target[:,i] + self.voltage_at_target[:, i+2]) / (2*r*10**(-2) * del_phi*cos_theta)
        self.curr_density_phi = self.cond_vec[len(self.cond_vec)-1]*self.curr_density_phi/10

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_norm_calculated = True
        self.curr_density_r = curr_density.data

        # print(self.curr_density_phi[:,0])
        # print(self.curr_density_theta[:,0])
        # print(">>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")

        self.curr_density_norm = np.sqrt(self.curr_density_phi**2+self.curr_density_theta**2+self.curr_density_r**2)
        return self.curr_density_norm

    def plot_elec_pttrn(self, J=None, x_lim=None, y_lim=None, annotate=True, fname=None, show=True):
        elec_lst, ground_elec, elec_radius = self.approx_rectangular(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec)
        x_elec, y_elec = self.r_max * 10 ** 2 * np.cos(elec_lst[:, 0]) * np.cos(elec_lst[:, 1]), self.r_max * 10 ** 2 * np.cos(elec_lst[:, 0]) * np.sin(elec_lst[:, 1])
        if J is None:
            J=np.ones(len(elec_lst))
        J = J*elec_radius**2*np.pi

        plt.scatter(x_elec, y_elec, alpha=0.7, s=100 * np.array(np.abs(J) > np.max(np.abs(J)) * 0.01), c=np.array((J > 0), dtype="float32"), cmap='seismic')
        I_bottom_wall_Z = J
        if annotate is True:
            for i, txt in enumerate(I_bottom_wall_Z):
                if np.abs(J[i]) > np.max(np.abs(J)) * 0.01:
                    plt.annotate(str(round(txt,2)), (x_elec[i], y_elec[i]))
        plt.xlabel("X-direction (cm)", fontsize='16')
        plt.ylabel("Y-direction (cm)", fontsize='16')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Electrode Pattern", fontsize='18')
        plt.tight_layout()
        plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()

    def plot_given_curr_norm_1d(self, r, curr_density, length=1, angle=0, normalize=False, fname=None, num_points=500, show=True):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = curr_density.flatten()
        data_flatten = data_flatten[idx]
        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]

        lin_coord_x, lin_coord_y = np.arange(0, length+self.eps, length/num_points)*np.cos(angle), np.arange(0, length+self.eps, length/num_points)*np.sin(angle)
        lin_coord_x, lin_coord_y = max_x+lin_coord_x, max_y+lin_coord_y
        data_projected = np.zeros((len(lin_coord_x),))
        spacing = length / num_points


        for i in range(len(lin_coord_y)):
            data_projected[i] = np.mean(data_flatten[np.where(np.square(x - lin_coord_x[i]) + np.square(y - lin_coord_y[i]) <= spacing**2)])

        if normalize:
            data_projected = data_projected/np.max(np.abs(data_projected))

        plt.plot(np.arange(0, length+self.eps, length/num_points)*10, data_projected)

        # labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        # labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        # labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        # labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        # labels_y = np.flip(np.array(labels_y))
        # x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        # y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        # plt.xticks(x_ticks, labels_x, fontsize='13')
        # plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xticks(fontsize='13')
        plt.yticks(fontsize='13')
        plt.xlabel('Distance from Centre (mm)', fontsize='17')
        plt.ylabel('$\|J\|$ (mA/cm$^2$)', fontsize='17')
        plt.title('Current Density Norm ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
        return [max_x, max_y], data_projected

    def plot_curr_norm_1d(self, r, J=None, length=1, angle=0, fname=None, normalize=True, num_points=500, show=True):
        if J is None:
           if self.curr_norm_calculated is False:
                self.calc_Jnorm(r)
        else:
            self.calc_Jnorm(r,J)
        return self.plot_given_curr_norm_1d(r=r, curr_density=self.curr_density_norm, length=length, angle=angle, fname=fname, normalize=normalize, num_points=num_points, show=show)
    def plot_given_curr_alldir_1d(self, r, voltage, length=1, angle=0, normalize=False, fname=None, num_points=500, show=True):

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half

        idx = spherical_map_flatten[:, 1] <= 0
        x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = voltage.flatten()
        data_flatten = data_flatten[idx]
        max_idx = np.argmax(np.abs(data_flatten))
        max_x, max_y = x[max_idx], y[max_idx]

        lin_coord_x, lin_coord_y = np.arange(0, length+self.eps, length/num_points)*np.cos(angle), np.arange(0, length+self.eps, length/num_points)*np.sin(angle)
        lin_coord_x, lin_coord_y = max_x+lin_coord_x, max_y+lin_coord_y
        data_projected = np.zeros((len(lin_coord_x),))
        spacing = length / num_points


        for i in range(len(lin_coord_y)):
            data_projected[i] = np.mean(data_flatten[np.where(np.square(x - lin_coord_x[i]) + np.square(y - lin_coord_y[i]) <= spacing**2)])

        if normalize:
            data_projected = data_projected/np.max(np.abs(data_projected))

        plt.plot(np.arange(0, length+self.eps, length/num_points)*10, data_projected)

        # labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
        # labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        # labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
        # labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        # labels_y = np.flip(np.array(labels_y))
        # x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
        # y_ticks = np.linspace(0, len(y_discretize), len(labels_y))

        # plt.xticks(x_ticks, labels_x, fontsize='13')
        # plt.yticks(y_ticks, labels_y, fontsize='13')
        plt.xticks(fontsize='13')
        plt.yticks(fontsize='13')
        plt.xlabel('Distance from Centre (mm)', fontsize='17')
        plt.ylabel('$\|J\|$ (mA/cm$^2$)', fontsize='17')
        plt.title('Current Density Norm ('+str(round(self.r_max*10**2-r,2))+" cm)", fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        if show:
            plt.show()
        else:
            plt.clf()
            plt.cla()
        return [max_x, max_y], data_projected

    def plot_curr_norm_1d(self, r, J=None, length=1, angle=0, fname=None, normalize=True, num_points=500, show=True):
        if J is None:
           if self.curr_norm_calculated is False:
                self.calc_Jnorm(r)
        else:
            self.calc_Jnorm(r,J)
        return self.plot_given_curr_norm_1d(r=r, curr_density=self.curr_density_norm, length=length, angle=angle, fname=fname, normalize=normalize, num_points=num_points, show=show)

    def SparsePlaceVanilla_ENorm(self, Jdes, Jsafety, Af=None, Ac=None, beta=0.001, J_return=None, area_elec=None):

        Af_r, Af_theta, Af_phi = Af[0], Af[1], Af[2]
        Ac_r, Ac_theta, Ac_phi = Ac[0], Ac[1], Ac[2]

        if area_elec is None:
            area_elec = np.ones(Af_r.shape[1])
        I = cvx.Variable(Af_r.shape[1])

        if J_return is None:
            constraints = [Af_r @ I == Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(I) <= beta]
        else:
            ones = np.zeros((1, Af.shape[1]))
            ones[len(ones) - 1] = 1
            constraints = [Af_r @ I == Jdes, ones @ I == J_return, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(cvx.multiply(I, area_elec)) <= beta]

        obj = cvx.Minimize(cvx.atoms.norm(cvx.vstack([Ac_r, Ac_theta, Ac_phi]) @ I))  # +beta*cvx.atoms.norm1(I)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.MOSEK)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def SparsePlaceHinge_ENorm(self, Jdes, Jsafety, Jtol, Af=None, Ac=None, beta=0.001, J_return=None, area_elec=None):

        Af_r, Af_theta, Af_phi = Af[0], Af[1], Af[2]
        Ac_r, Ac_theta, Ac_phi = Ac[0], Ac[1], Ac[2]

        if area_elec is None:
            area_elec = np.ones(Af_r.shape[1])

        I = cvx.Variable(Af_r.shape[1])
        if J_return is None:
            constraints = [Af_r @ I == Jdes, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(I) <= beta]
        else:
            ones = np.zeros((1, Af.shape[1]))
            ones[len(ones) - 1] = 1
            constraints = [Af_r @ I == Jdes, ones @ I == J_return, cvx.sum(I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(cvx.multiply(I, area_elec)) <= beta]
        J_norm = cvx.power(Ac_r@I, 2)+cvx.power(Ac_theta@I, 2)+cvx.power(Ac_phi@I,2)
        obj = cvx.Minimize(cvx.sum(cvx.atoms.maximum(0,J_norm- Jtol**2*np.ones((Ac_r.shape[0])))))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.MOSEK) #max_iters  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def run_SparsePlaceVanilla_ENorm(self, per, Jdes, Jsafety, focus_points, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):
        print(">>>>>>>>>Running Vanilla Loss SparsePlace<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_sparseplace_vanilla = self.SparsePlaceVanilla_ENorm(Jdes=Jdes, Jsafety=Jsafety, Af=[self.Af, self.Af_theta, self.Af_phi], Ac=[self.Ac, self.Ac_theta, self.Ac_phi], beta=beta)
        else:
            self.J_sparseplace_vanilla = self.SparsePlaceVanilla_ENorm(Jdes=Jdes, Jsafety=Jsafety, Af=Af, Ac=Ac, beta=beta)

        if self.J_sparseplace_vanilla is None:
            return -1, -1
        else:
            curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_vanilla)
            print("Error Factor:", Jdes / np.max(curr_density))
            self.J_sparseplace_vanilla = Jdes / np.max(curr_density) * self.J_sparseplace_vanilla
            if print_elec_pattern:
                self.plot_elec_pttrn(self.J_sparseplace_vanilla, fname=fname_elec, x_lim=np.array((-0.2,0.2)), y_lim=np.array((-0.2,0.2)))
            if plot is True:
                max_x, max_y = self.plot_curr_density(r=np.min(focus_points[:,0]),  J=self.J_sparseplace_vanilla, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_density, abs=False)
                print("Coordinates of the centre:", max_x, max_y, 'for Jdes:', Jdes)
            if density_thresh is None:
                area = self.area_above_x_norm(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_vanilla, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_area)
            else:
                area = self.area_above_x_norm(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_vanilla, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), density_thresh=density_thresh, fname=fname_area)

        return self.J_sparseplace_vanilla, area

    def run_SparsePlaceHinge_ENorm(self, per, Jdes, Jsafety, focus_points, Jtol, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):

        print(">>>>>>>>>Running Hinge Loss SparsePlace<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_sparseplace_hinge = self.SparsePlaceHinge_ENorm(Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol, Af=[self.Af, self.Af_theta, self.Af_phi], Ac=[self.Ac,self.Ac_theta, self.Ac_phi], beta=beta)
        else:
            self.J_sparseplace_hinge = self.SparsePlaceHinge_ENorm(Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol, Af=Af, Ac=Ac, beta=beta)

        if self.J_sparseplace_hinge is None:
            return -1,-1
        else:
            curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_hinge)
            print("Error Factor:", Jdes / np.max(curr_density))
            self.J_sparseplace_hinge = Jdes / np.max(curr_density) * self.J_sparseplace_hinge
            if print_elec_pattern:
                self.plot_elec_pttrn(self.J_sparseplace_hinge, fname=fname_elec, x_lim=np.array((-0.2, 0.2)), y_lim=np.array((-0.2, 0.2)))
            if plot is True:
                max_x, max_y = self.plot_curr_density(r=np.min(focus_points[:,0]), J=self.J_sparseplace_hinge, fname=fname_density, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), abs=False)
                print("Coordinates of the centre:", max_x, max_y, 'for Jdes:', Jdes)
            if density_thresh is None:
                area = self.area_above_x_norm(r=np.min(focus_points[:,0]), per=per, plot=plot, J=self.J_sparseplace_hinge,x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_area)
            else:
                area = self.area_above_x_norm(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_hinge, x_limit=np.array((-1.5,1.5)), y_limit=np.array((-1.5,1.5)), fname=fname_area, density_thresh=density_thresh)

        return self.J_sparseplace_hinge, area

    def area_above_x_norm(self, per, r, plot=False, J=None, x_limit=None, y_limit=None, fname=None, density_thresh=None, print_area=True):
        if self.curr_norm_calculated is False or J is not None:
            self.calc_Jnorm(r=r, J=J)
        area = self.area_above_x_curr_norm(per=per, r=r, plot=plot, curr_norm=self.curr_density_norm, x_limit=x_limit, y_limit=y_limit, fname=fname, density_thresh=density_thresh, print_area=print_area)
        return area

    def area_above_x_curr_norm(self, per, r, curr_norm, plot=False, print_area=True, x_limit=None, y_limit=None, fname=None, density_thresh=None):

        max_density = np.max(curr_norm)
        if density_thresh is None:
            density_thresh = max_density * per
        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))
        idx = spherical_map_flatten[:, 1] <= 0
        spherical_map_flatten = spherical_map_flatten[idx]
        data_flatten = curr_norm.flatten()
        data_flatten = data_flatten[idx]
        points_above_thresh = spherical_map_flatten[data_flatten > density_thresh]
        del_theta = np.abs(self.lat_theta[0] - self.lat_theta[1])
        del_phi = np.abs(self.long_phi[0] - self.long_phi[1])
        area = r**2 * del_theta * del_phi * np.sum(np.cos(points_above_thresh[:, 1]))
        if print_area:
            print("Area above " + str(int(per * 100)) + " of the max:", area, "cm^2")
        if plot is True:
            idx = spherical_map_flatten[:, 1] <= 0
            x, y = r * np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), r * np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
            # data_flatten = curr_density.flatten()
            idx_1 = data_flatten > density_thresh
            idx_2 = data_flatten <= density_thresh
            data_flatten[idx_1] = 1
            data_flatten[idx_2] = 0
            data_flatten = data_flatten[idx]

            if x_limit is None:
                x_limit = np.array((np.min(x), np.max(x)))
            if y_limit is None:
                y_limit = np.array((np.min(y), np.max(y)))

            x_discretize = np.arange(x_limit[0], x_limit[1] + self.eps, (x_limit[1] - x_limit[0]) / 100)
            y_discretize = np.arange(y_limit[0], y_limit[1] + self.eps, (y_limit[1] - y_limit[0]) / 100)

            spacing_x = (x_limit[1] - x_limit[0]) / 100
            spacing_y = (y_limit[1] - y_limit[0]) / 100
            data_projected = np.zeros((len(x_discretize), len(y_discretize)))
            for i in range(len(x_discretize)):
                for j in range(len(y_discretize)):
                    data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
            data_projected = np.transpose(data_projected)
            data_projected = np.flip(data_projected, axis=0)
            data_projected = data_projected>0.5
            sns.heatmap(np.abs(data_projected), cmap="jet")
            labels_x = np.linspace(x_limit[0]*10, x_limit[1]*10, 11)
            labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
            labels_y = np.linspace(y_limit[0]*10, y_limit[1]*10, 11)
            labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
            labels_y = np.flip(np.array(labels_y))
            x_ticks = np.linspace(0, len(x_discretize), len(labels_x))
            y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
            plt.xticks(x_ticks, labels_x, fontsize='13')
            plt.yticks(y_ticks, labels_y, fontsize='13')
            plt.xlabel('x-axis (mm)', fontsize='17')
            plt.ylabel('y-axis (mm)', fontsize='17')
            plt.title('Region Activated (' + str(round(self.r_max * 10 ** (2) - r,2)) +' cm)', fontsize='21')
            plt.tight_layout()
            plt.savefig(fname)
            plt.show()
        return area

    def __call__(self, Jdes, Jsafety, focus_points, per=0.8,optimizer =1, Jtol=None, alpha_E=1e9, alpha_I=None, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):

        if optimizer == 1:
            J_elec, area = self.run_SparsePlaceVanilla(per=per,Jdes=Jdes, Jsafety=Jsafety, focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        elif optimizer == 2:
            J_elec, area = self.run_SparsePlaceHinge(per=per,Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol,focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        elif optimizer == 3:
            J_elec, area = self.run_constrained_Maximiztion(per=per,alpha_E=alpha_E, Jsafety=Jsafety, alpha_I=alpha_I,focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, density_thresh=density_thresh)
        elif optimizer == 4:
            J_elec, area = self.run_SparsePlaceVanilla_ENorm(per=per,Jdes=Jdes, Jsafety=Jsafety, focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        elif optimizer == 5:
            J_elec, area = self.run_SparsePlaceHinge_ENorm(per=per,Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol,focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        return J_elec, area

@ray.remote
class SparsePlaceNHP(SparsePlaceSpherical):

    def __init__(self, cond_vec, radius_vec, num_elec, elec_radius, elec_spacing, elec_patch_angle = 75./180.0*np.pi,via_size=None, max_l=300, spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None, big_elec_pos=None, big_elec_radius=None):

        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten
        self.num_elec = num_elec
        self.custom_grid = custom_grid

        if self.custom_grid is True:
            self.theta_elec = theta_elec
            self.phi_elec = phi_elec

        self.elec_spacing = elec_spacing * 10**(-2)
        self.elec_patch_angle = elec_patch_angle
        self.elec_radius = elec_radius * 10**(-2)
        self.big_elec_radius = big_elec_radius
        self.big_elec_pos = big_elec_pos
        if via_size is None:
            self.via_size = None
        else:
            self.via_size = via_size*10**(-2)
        self.curr_density_calculated = False

    def forward_model(self, focus_points, cancel_points, print_elec_pattern=False, save=True, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy"):

        points = np.vstack((np.array(focus_points), np.array(cancel_points)))
        r = np.unique(points[:,0])

        ## Generate a pseudo-uniform sampling of electrode points
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2,None

        if print_elec_pattern:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=np.ones(len(elec_lst)), via_size=via_size)
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()

        ## Calculating the Transfer Function
        self.Af = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac = np.zeros((len(cancel_points), len(elec_lst)))

        previous_len_focus = 0
        previous_len_cancel = 0


        for j in range(len(r)):
            _, tau_Jr, _ = self.calc_tauL(r[j])
            tau_Jr= np.hstack((np.array((0,)), tau_Jr))

            idx_focus = np.abs(focus_points[:, 0] - r[j]) < self.eps
            focus_points_subsampled = focus_points[idx_focus]

            idx_cancel = np.abs(cancel_points[:, 0] - r[j]) < self.eps
            cancel_points_subsampled = cancel_points[idx_cancel]

            ## Calculate the Forward Model
            for i in range(len(elec_lst)):

                start_time = time.time()
                elec_pos = np.vstack((elec_lst[i], ground_elec))
                elec_radii = self.elec_radius*np.ones(len(elec_pos))*10**(2)
                inj_curr = np.array((1, -1 * (elec_radius[i] ** 2-via_size[i]**2) / ((self.elec_radius*10**2)**2-via_size[0]**2)))

                # inj_curr = np.array((1,-1))
                self.electrode_sampling(elec_pos, elec_radii, inj_curr, np.array([self.via_size, self.via_size]))
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)

                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                # elec_grid.plot()
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Multipying the transfer function
                curr_density_coeff = np.zeros(coeff_array.shape)
                curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
                curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
                curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

                curr_density = curr_density_coeff.expand(grid='DH')
                curr_density_data = curr_density.data

                ## Focus Points
                for ii in range(len(focus_points_subsampled)):
                    idx1 = np.searchsorted(self.lat_theta, focus_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, focus_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta) - idx1, idx2]
                    self.Af[ii+j*previous_len_focus,i] = current_at_target

                ## Cancel Points
                for ii in range(len(cancel_points_subsampled)):

                    idx1 = np.searchsorted(self.lat_theta, cancel_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, cancel_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta)-idx1, idx2]
                    self.Ac[ii+j*previous_len_cancel,i]= current_at_target

                print("Electrode Remaining:", len(elec_lst)-i-1, "Time Taken:", time.time()-start_time)

            previous_len_focus = previous_len_focus+len(focus_points_subsampled)
            previous_len_cancel = previous_len_cancel+len(cancel_points_subsampled)
        if save:
            np.save(save_title_focus, self.Af)
            np.save(save_title_cancel, self.Ac)
        return self.Af, self.Ac

    def get_Af_and_Ac(self):
        return self.Af, self.Ac

    def calc_voltage(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        tau_V, _, _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.volatge_calculated = True
        self.voltage_at_target = curr_density.data*10
        return self.voltage_at_target

    def calc_curr_density(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla, via_size=via_size)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J, via_size=via_size)

        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        _, tau_Jr, _ = self.calc_tauL(r)
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_density_calculated = True
        self.curr_density_at_target = curr_density.data
        return self.curr_density_at_target

    def plot_elec_pttrn(self, J, x_lim=None, y_lim=None, fname=None):

        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        J = np.multiply(J, ((elec_radius.reshape(J.shape)) ** 2-(via_size.reshape(J.shape))**2) * np.pi)
        self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J, via_size=via_size)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        grid_electrode.plot(colorbar='right')
        data = self.electrode_spherical_map

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] >= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 100)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 100)

        spacing_x = (x_lim[1] - x_lim[0]) / 100
        spacing_y = (y_lim[1] - y_lim[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")

        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('x-axis (cm)', fontsize='19')
        plt.ylabel('y-axis (cm)', fontsize='19')
        plt.title('Electrode Pattern', fontsize='21')
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

    def minimize_linfty(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm_inf(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def minimize_l2(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def minimize_l1(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def SparsePlaceVanilla(self, Jdes, Jsafety, Af=None, Ac=None, beta=0.001, J_return=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        if hasattr(Jsafety, "__len__"):
            constraints = [Af @ I == Jdes, cvx.sum(area_elec @ I) == 0, cvx.atoms.abs(I) <= Jsafety, cvx.atoms.norm1(cvx.multiply(area_elec.flatten(),I)) <= beta]
        else:
            if J_return is None:
                constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(cvx.multiply(area_elec.flatten(),I))<=beta]
            else:
                ones = np.zeros((Af.shape[1]))
                ones[len(ones) - 1] = 1
                zeros = 1 - ones
                ones = ones.reshape(1, -1)

                constraints = [Af @ I == Jdes, cvx.atoms.abs(ones@I +J_return)<= 0.1, cvx.sum(area_elec @ I) == 0, cvx.atoms.norm_inf(cvx.multiply(zeros, I)) <= Jsafety, cvx.atoms.norm1(cvx.multiply(area_elec.flatten(), I)) <= beta]
        obj = cvx.Minimize(cvx.atoms.norm(Ac@I)) #+beta*cvx.atoms.norm1(I)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def SparsePlaceHinge(self, Jdes, Jsafety, Jtol, Af=None, Ac=None, beta=0.001, J_return=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None


        elec_radius = elec_radius.reshape(1,-1)
        area_elec = (elec_radius**2-via_size**2)*np.pi
        I = cvx.Variable(Af.shape[1])
        if hasattr(Jsafety, "__len__"):
            constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.abs(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        else:
            if J_return is None:
                constraints = [Af @ I == Jdes, cvx.sum(area_elec @ I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(I) <= beta]
            else:
                ones = np.zeros((Af.shape[1]))
                ones[len(ones) - 1] = 1
                zeros = 1 - ones
                ones = ones.reshape(1,-1)

                constraints = [Af @ I == Jdes, cvx.atoms.abs(ones@I +J_return)<= 0.1, cvx.sum(area_elec @ I) == 0, cvx.atoms.norm_inf(cvx.multiply(zeros, I)) <= Jsafety, cvx.atoms.norm1(cvx.multiply(zeros, I)) <= beta]

        obj = cvx.Minimize(cvx.sum(cvx.atoms.maximum(0, Ac@I-Jtol*np.ones((Ac.shape[0]))))+cvx.sum(cvx.atoms.maximum(0, -Ac@I-Jtol*np.ones((Ac.shape[0])))))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=300)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("constraint", np.matmul(Af.reshape(1,-1), I.value))
        print('constraint:', np.sum(np.multiply(I.value, area_elec)))
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def run_diffused(self, per, Jdes, focus_points, Af=None, print_elec_pattern=False,plot=True, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):
        print(">>>>>>>>>Running Hinge Loss SparsePlace<<<<<<<<<<<")

        self.J_l2 = self.minimize_l2(Jdes=Jdes, Af=Af)
        self.J_linfty = self.minimize_linfty(Jdes=Jdes, Af=Af)
        self.J_l1 = self.minimize_l1(Jdes=Jdes, Af=Af)


        # curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_hinge)
        # print("Error Factor:", Jdes / np.max(curr_density))
        #self.J_sparseplace_hinge = Jdes / np.max(curr_density) * self.J_sparseplace_hinge
        if print_elec_pattern:
            self.plot_elec_pttrn(self.J_l2, fname=fname_elec+"l2.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
            self.plot_elec_pttrn(self.J_l1, fname=fname_elec + "l1.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
            self.plot_elec_pttrn(self.J_linfty, fname=fname_elec + "linfty.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
        if plot is True:
            self.plot_curr_density(r=np.min(focus_points[:,0]), J=self.J_l2, fname=fname_density+"_l2.png", x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), abs=False)
            self.plot_curr_density(r=np.min(focus_points[:, 0]), J=self.J_l1, fname=fname_density+"_l2.png", x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), abs=False)
            self.plot_curr_density(r=np.min(focus_points[:, 0]), J=self.J_linfty, fname=fname_density+"_l2.png", x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), abs=False)
        if density_thresh is None:
            area_l2 = self.area_above_x(r=np.min(focus_points[:,0]), per=per, plot=plot, J=self.J_l2,x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area+"_l2.png")
            area_l2 = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_l2, x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), fname=fname_area + "_l2.png")
        else:
            area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_hinge, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area, density_thresh=density_thresh)

        return self.J_l2,self.J_linfty,self.J_l1, area

    def run_SparsePlaceVanilla(self, per, Jdes, Jsafety, focus_points,J_return=None, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):
        print(">>>>>>>>>Running Vanilla Loss SparsePlace<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_sparseplace_vanilla = self.SparsePlaceVanilla(Jdes=Jdes, J_return=J_return, Jsafety=Jsafety, Af=self.Af, Ac=self.Ac, beta=beta)
        else:
            self.J_sparseplace_vanilla = self.SparsePlaceVanilla(Jdes=Jdes, J_return=J_return, Jsafety=Jsafety, Af=Af, Ac=Ac, beta=beta)

        if self.J_sparseplace_vanilla is None:
            return -1, -1
        else:
            curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_vanilla)
            print("Error Factor:", Jdes / np.max(curr_density))
            self.J_sparseplace_vanilla = Jdes / np.max(curr_density) * self.J_sparseplace_vanilla
            if print_elec_pattern:
                self.plot_elec_pttrn(self.J_sparseplace_vanilla, fname=fname_elec, x_lim=np.array((-1.2,1.2)), y_lim=np.array((-0.7,0.7)))
            if plot is True:
                max_x, max_y = self.plot_curr_density(r=np.min(focus_points[:,0]),  J=self.J_sparseplace_vanilla, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_density, abs=False)
            else:
                max_x, max_y = self.get_max_location(r=np.min(focus_points[:,0]), J=self.J_sparseplace_hinge)
            print("Coordinates of the centre:", max_x, max_y, 'for Jdes:', Jdes)
            if density_thresh is None:
                area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_vanilla, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area)
            else:
                area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_vanilla, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), density_thresh=density_thresh, fname=fname_area)

        return self.J_sparseplace_vanilla, area, [max_x, max_y]

    def run_constrained_Maximiztion(self, per, alpha_I, Jsafety, focus_points,alpha_E=1e9, cancel_points=None, Af=None, Ac=None, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, density_thresh=None):
        print(">>>>>>>>>Running Constrained Maximization<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_const_max = self.constrained_Maximization(alpha_I=alpha_I, alpha_E=alpha_E, Jsafety=Jsafety, Af=self.Af, Ac=self.Ac)
        else:
            self.J_const_max = self.constrained_Maximization(alpha_I=alpha_I, alpha_E=alpha_E, Jsafety=Jsafety, Af=Af, Ac=Ac)
        self.plot_elec_pttrn(self.J_const_max, x_lim=np.array((-0.2,0.2)), y_lim=np.array((-0.2,0.2)))
        # if plot is True:
        self.plot_curr_density(r=np.min(focus_points[:,0]),  J=self.J_const_max)
        if density_thresh is None:
            area = self.area_above_x(r=np.min(focus_points[:,0]), per=per, plot=True, J=self.J_const_max, x_limit=np.array((-0.2,0.2)), y_limit=np.array((-0.2,0.2)), abs=False)
        else:
            area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=True, J=self.J_const_max, x_limit=np.array((-0.2, 0.2)), y_limit=np.array((-0.2, 0.2)), density_thresh=density_thresh)
        return self.J_const_max, area

    def run_SparsePlaceHinge(self, per, Jdes, Jsafety, focus_points, Jtol, J_return=None, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):

        print(">>>>>>>>>Running Hinge Loss SparsePlace<<<<<<<<<<<")
        if Af is None or Ac is None:
            if cancel_points is None:
                centre = focus_points.reshape((-1,))
                offset = offset
                thickness = thickness
                cancel_points = self.uniform_ring_sample(centre, offset, thickness)
            self.forward_model(focus_points=focus_points, cancel_points=cancel_points, save_title_cancel=save_title_cancel, save_title_focus=save_title_focus, print_elec_pattern=print_elec_pattern)
            self.J_sparseplace_hinge = self.SparsePlaceHinge(Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol,J_return=J_return, Af=self.Af, Ac=self.Ac, beta=beta)
        else:
            self.J_sparseplace_hinge = self.SparsePlaceHinge(Jdes=Jdes, Jsafety=Jsafety, Jtol=Jtol, J_return=J_return, Af=Af, Ac=Ac, beta=beta)

        if self.J_sparseplace_hinge is None:
            return -1,-1
        else:
            curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_hinge)
            print("Error Factor:", Jdes / np.max(curr_density))
            self.J_sparseplace_hinge = Jdes / np.max(curr_density) * self.J_sparseplace_hinge
            if print_elec_pattern:
                self.plot_elec_pttrn(self.J_sparseplace_hinge, fname=fname_elec, x_lim=np.array((-1.4, 1.4)), y_lim=np.array((-0.7, 0.7)))
            if plot is True:
                max_x, max_y = self.plot_curr_density(r=np.min(focus_points[:,0]), J=self.J_sparseplace_hinge, fname=fname_density, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), abs=False)
            else:
                max_x, max_y = self.get_max_location(r=np.min(focus_points[:,0]), J=self.J_sparseplace_hinge)

            print("Coordinates of the centre:", max_x, max_y, 'for Jdes:', Jdes)
            if density_thresh is None:
                area = self.area_above_x(r=np.min(focus_points[:,0]), per=per, plot=plot, J=self.J_sparseplace_hinge,x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area)
            else:
                area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_hinge, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area, density_thresh=density_thresh)

        return self.J_sparseplace_hinge, area, [max_x, max_y]

    def __call__(self, Jdes, Jsafety, focus_points, J_return=None, per=0.8,optimizer =1, Jtol=None, alpha_E=1e9, alpha_I=None, cancel_points=None, Af=None, Ac=None, beta=0.001, print_elec_pattern=False, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy",plot=True, offset=None, thickness=None, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):

        if optimizer == 1:
            J_elec, area, loc = self.run_SparsePlaceVanilla(per=per,Jdes=Jdes,J_return=J_return, Jsafety=Jsafety, focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        elif optimizer == 2:
            J_elec, area, loc = self.run_SparsePlaceHinge(per=per,Jdes=Jdes, Jsafety=Jsafety,J_return=J_return, Jtol=Jtol,focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, beta=beta, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, fname_density=fname_density, fname_area=fname_area, fname_elec=fname_elec, density_thresh=density_thresh)
        elif optimizer == 3:
            J_elec, area = self.run_constrained_Maximiztion(per=per,alpha_E=alpha_E, Jsafety=Jsafety, alpha_I=alpha_I,focus_points=focus_points, cancel_points=cancel_points, Af=Af, Ac=Ac, print_elec_pattern=print_elec_pattern, save_title_focus=save_title_focus, save_title_cancel=save_title_cancel,plot=plot, offset=offset, thickness=thickness, density_thresh=density_thresh)

        return J_elec, area, loc

@ray.remote
class SparsePlaceRodent(SparsePlaceSpherical):

    def __init__(self, cond_vec, radius_vec, num_elec, elec_radius, elec_spacing,via_size=None, max_l=300, spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None):

        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten
        self.num_elec = num_elec
        self.custom_grid = custom_grid

        if self.custom_grid is True:
            self.theta_elec = theta_elec
            self.phi_elec = phi_elec

        self.elec_spacing = elec_spacing * 10**(-2)
        self.elec_radius = elec_radius * 10**(-2)
        self.big_elec_radius = None
        self.big_elec_pos = None
        if via_size is None:
            self.via_size = None
        else:
            self.via_size = via_size*10**(-2)
        self.curr_density_calculated = False

    def forward_model(self, focus_points, cancel_points, print_elec_pattern=False, save=True, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy"):

        points = np.vstack((np.array(focus_points), np.array(cancel_points)))
        r = np.unique(points[:,0])

        ## Generate a pseudo-uniform sampling of electrode points
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2,None

        if print_elec_pattern:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=np.ones(len(elec_lst)), via_size=via_size)
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()

        ## Calculating the Transfer Function
        self.Af = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac = np.zeros((len(cancel_points), len(elec_lst)))

        previous_len_focus = 0
        previous_len_cancel = 0


        for j in range(len(r)):
            _, tau_Jr, _ = self.calc_tauL(r[j])
            tau_Jr= np.hstack((np.array((0,)), tau_Jr))

            idx_focus = np.abs(focus_points[:, 0] - r[j]) < self.eps
            focus_points_subsampled = focus_points[idx_focus]

            idx_cancel = np.abs(cancel_points[:, 0] - r[j]) < self.eps
            cancel_points_subsampled = cancel_points[idx_cancel]

            ## Calculate the Forward Model
            for i in range(len(elec_lst)):

                start_time = time.time()
                elec_pos = np.vstack((elec_lst[i], ground_elec))
                elec_radii = self.elec_radius*np.ones(len(elec_pos))*10**(2)
                inj_curr = np.array((1, -1 * (elec_radius[i] ** 2-via_size[i]**2) / ((self.elec_radius*10**2)**2-via_size[0]**2)))

                # inj_curr = np.array((1,-1))
                self.electrode_sampling(elec_pos, elec_radii, inj_curr, np.array([self.via_size, self.via_size]))
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)

                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                # elec_grid.plot()
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Multipying the transfer function
                curr_density_coeff = np.zeros(coeff_array.shape)
                curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
                curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
                curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

                curr_density = curr_density_coeff.expand(grid='DH')
                curr_density_data = curr_density.data

                ## Focus Points
                for ii in range(len(focus_points_subsampled)):
                    idx1 = np.searchsorted(self.lat_theta, focus_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, focus_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta) - idx1, idx2]
                    self.Af[ii+j*previous_len_focus,i] = current_at_target

                ## Cancel Points
                for ii in range(len(cancel_points_subsampled)):

                    idx1 = np.searchsorted(self.lat_theta, cancel_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, cancel_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta)-idx1, idx2]
                    self.Ac[ii+j*previous_len_cancel,i]= current_at_target

                print("Electrode Remaining:", len(elec_lst)-i-1, "Time Taken:", time.time()-start_time)

            previous_len_focus = previous_len_focus+len(focus_points_subsampled)
            previous_len_cancel = previous_len_cancel+len(cancel_points_subsampled)
        if save:
            np.save(save_title_focus, self.Af)
            np.save(save_title_cancel, self.Ac)
        return self.Af, self.Ac

    def get_Af_and_Ac(self):
        return self.Af, self.Ac

    def calc_voltage(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        tau_V, _, _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.volatge_calculated = True
        self.voltage_at_target = curr_density.data*10
        return self.voltage_at_target

    def calc_curr_density(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla, via_size=via_size)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J, via_size=via_size)

        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        _, tau_Jr, _ = self.calc_tauL(r)
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_density_calculated = True
        self.curr_density_at_target = curr_density.data
        return self.curr_density_at_target

    def plot_elec_pttrn(self, J, x_lim=None, y_lim=None, fname=None):

        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None
        J = np.multiply(J, ((elec_radius.reshape(J.shape)) ** 2-(via_size.reshape(J.shape))**2) * np.pi)

        self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J, via_size=via_size)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        grid_electrode.plot(colorbar='right')
        data = self.electrode_spherical_map

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] >= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 130)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 130)

        spacing_x = (x_lim[1] - x_lim[0]) / 130
        spacing_y = (y_lim[1] - y_lim[0]) / 130
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")

        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i]*10,1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i]*10,1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('x-axis (mm)', fontsize='19')
        plt.ylabel('y-axis (mm)', fontsize='19')
        plt.title('Electrode Pattern', fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def minimize_linfty(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm_inf(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def minimize_l2(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def minimize_l1(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def SparsePlaceVanilla(self, Jdes, Jsafety, Af=None, Ac=None, beta=0.001):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        if hasattr(Jsafety, "__len__"):
            constraints = [Af @ I == Jdes, cvx.sum(area_elec @ I) == 0, cvx.atoms.abs(I) <= Jsafety, cvx.atoms.norm1(I) <= beta]
        else:
            constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        obj = cvx.Minimize(cvx.atoms.norm(Ac@I)) #+beta*cvx.atoms.norm1(I)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def SparsePlaceHinge(self, Jdes, Jsafety, Jtol, Af=None, Ac=None, beta=0.001):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None


        elec_radius = elec_radius.reshape(1,-1)
        area_elec = (elec_radius**2-via_size**2)*np.pi
        I = cvx.Variable(Af.shape[1])
        if hasattr(Jsafety, "__len__"):
            constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.abs(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        else:
            constraints = [Af @ I == Jdes, cvx.sum(area_elec @ I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(I) <= beta]
        obj = cvx.Minimize(cvx.sum(cvx.atoms.maximum(0, Ac@I-Jtol*np.ones((Ac.shape[0]))))+cvx.sum(cvx.atoms.maximum(0, -Ac@I-Jtol*np.ones((Ac.shape[0])))))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=300)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("constraint", np.matmul(Af.reshape(1,-1), I.value))
        print('constraint:', np.sum(np.multiply(I.value, area_elec)))
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def run_diffused(self, per, Jdes, focus_points, Af=None, print_elec_pattern=False,plot=True, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):
        print(">>>>>>>>>Running Hinge Loss SparsePlace<<<<<<<<<<<")

        self.J_l2 = self.minimize_l2(Jdes=Jdes, Af=Af)
        self.J_linfty = self.minimize_linfty(Jdes=Jdes, Af=Af)
        self.J_l1 = self.minimize_l1(Jdes=Jdes, Af=Af)


        # curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_hinge)
        # print("Error Factor:", Jdes / np.max(curr_density))
        #self.J_sparseplace_hinge = Jdes / np.max(curr_density) * self.J_sparseplace_hinge
        if print_elec_pattern:
            self.plot_elec_pttrn(self.J_l2, fname=fname_elec+"l2.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
            self.plot_elec_pttrn(self.J_l1, fname=fname_elec + "l1.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
            self.plot_elec_pttrn(self.J_linfty, fname=fname_elec + "linfty.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
        if plot is True:
            self.plot_curr_density(r=np.min(focus_points[:,0]), J=self.J_l2, fname=fname_density+"_l2.png", x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), abs=False)
            self.plot_curr_density(r=np.min(focus_points[:, 0]), J=self.J_l1, fname=fname_density+"_l2.png", x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), abs=False)
            self.plot_curr_density(r=np.min(focus_points[:, 0]), J=self.J_linfty, fname=fname_density+"_l2.png", x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), abs=False)
        if density_thresh is None:
            area_l2 = self.area_above_x(r=np.min(focus_points[:,0]), per=per, plot=plot, J=self.J_l2,x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area+"_l2.png")
            area_l2 = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_l2, x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), fname=fname_area + "_l2.png")
        else:
            area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_hinge, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area, density_thresh=density_thresh)

        return self.J_l2,self.J_linfty,self.J_l1, area

class SparsePlaceNegativeElec(SparsePlaceSpherical):

    def __init__(self, cond_vec, radius_vec, patch_size, elec_radius, elec_spacing, max_l=300, spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None, neg_elec_pos=None, neg_elec_radius=None):

        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten
        self.custom_grid = custom_grid
        if self.custom_grid is False:
            self.patch_size = patch_size * 10**(-2)
            self.elec_spacing = elec_spacing * 10**(-2)
        else:
            self.theta_elec = theta_elec
            self.phi_elec = phi_elec
        self.elec_radius = elec_radius * 10**(-2)
        self.curr_density_calculated = False
        self.neg_elec_pos = neg_elec_pos
        self.neg_elec_radius=neg_elec_radius

    def forward_model(self, focus_points, cancel_points, print_elec_pattern=False, save=True, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy"):

        points = np.vstack((np.array(focus_points), np.array(cancel_points)))
        r = np.unique(points[:,0])

        ## Generate a pseudo-uniform sampling of electrode points
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole_extra(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**(2), patch_size=self.patch_size*10**(2), neg_elec_radius=self.neg_elec_radius, neg_elec_pos=self.neg_elec_pos)
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if print_elec_pattern:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=np.ones(len(elec_lst)))
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()


        ## Calculating the Transfer Function
        self.Af = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac = np.zeros((len(cancel_points), len(elec_lst)))

        previous_len_focus = 0
        previous_len_cancel = 0

        for j in range(len(r)):
            _, tau_Jr, _ = self.calc_tauL(r[j])
            tau_Jr= np.hstack((np.array((0,)), tau_Jr))

            idx_focus = np.abs(focus_points[:, 0] - r[j]) < self.eps
            focus_points_subsampled = focus_points[idx_focus]

            idx_cancel = np.abs(cancel_points[:, 0] - r[j]) < self.eps
            cancel_points_subsampled = cancel_points[idx_cancel]

            ## Calculate the Forward Model
            for i in range(len(elec_lst)):

                start_time = time.time()
                elec_pos = np.vstack((elec_lst[i], ground_elec))
                elec_radii = np.array((elec_radius[i], self.elec_radius*10**2))

                inj_curr = np.array((1,-1*(elec_radius[i]**2)/(self.elec_radius*10**2)**2))
                # print(inj_curr)
                self.electrode_sampling(elec_pos, elec_radii, inj_curr)
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)

                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                # elec_grid.plot()
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Multipying the transfer function
                curr_density_coeff = np.zeros(coeff_array.shape)
                curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
                curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
                curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

                curr_density = curr_density_coeff.expand(grid='DH')
                curr_density_data = curr_density.data

                ## Focus Points
                for ii in range(len(focus_points_subsampled)):
                    idx1 = np.searchsorted(self.lat_theta, focus_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, focus_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta) - idx1, idx2]
                    self.Af[ii+j*previous_len_focus,i] = current_at_target

                ## Cancel Points
                for ii in range(len(cancel_points_subsampled)):

                    idx1 = np.searchsorted(self.lat_theta, cancel_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, cancel_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta)-idx1, idx2]
                    self.Ac[ii+j*previous_len_cancel,i]= current_at_target

                print("Electrode Remaining:", len(elec_lst)-i-1, "Time Taken:", time.time()-start_time)

            previous_len_focus = previous_len_focus+len(focus_points_subsampled)
            previous_len_cancel = previous_len_cancel+len(cancel_points_subsampled)
        if save:
            np.save(save_title_focus, self.Af)
            np.save(save_title_cancel, self.Ac)
        return self.Af, self.Ac

    def calc_curr_density(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole_extra(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2),patch_size=self.patch_size * 10 ** (2), neg_elec_radius=self.neg_elec_radius, neg_elec_pos=self.neg_elec_pos)
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        _, tau_Jr, _ = self.calc_tauL(r)
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_density_calculated = True
        self.curr_density_at_target = curr_density.data
        return self.curr_density_at_target

    def SparsePlaceVanilla(self, Jdes, Jsafety, Af=None, Ac=None, beta=0.001):
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole_extra(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2),patch_size=self.patch_size * 10 ** (2), neg_elec_radius=self.neg_elec_radius, neg_elec_pos=self.neg_elec_pos)
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2
        elec_radius = elec_radius.reshape(1,-1)
        area_elec = elec_radius**2*np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        obj = cvx.Minimize(cvx.atoms.norm(Ac@I)) #+beta*cvx.atoms.norm1(I)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def SparsePlaceHinge(self, Jdes, Jsafety, Jtol, Af=None, Ac=None, beta=0.001):
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole_extra(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2),patch_size=self.patch_size * 10 ** (2), neg_elec_radius=self.neg_elec_radius, neg_elec_pos=self.neg_elec_pos)
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2
        elec_radius = elec_radius.reshape(1,-1)
        area_elec = elec_radius**2*np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        obj = cvx.Minimize(cvx.sum(cvx.atoms.maximum(0, Ac@I-Jtol*np.ones((Ac.shape[0]))))+cvx.sum(cvx.atoms.maximum(0, -Ac@I-Jtol*np.ones((Ac.shape[0])))))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=False, solver=cvx.ECOS, max_iters=300)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def plot_elec_pttrn(self, J, x_lim=None, y_lim=None, fname=None):
        if self.custom_grid is False:
            elec_lst, ground_elec, elec_radius = self.uniform_sampling_north_pole_extra(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** (2), patch_size=self.patch_size * 10 ** (2), neg_elec_radius=self.neg_elec_radius, neg_elec_pos=self.neg_elec_pos)
        else:
            elec_lst, ground_elec, elec_radius = np.hstack((self.theta_elec.reshape(-1,1), self.phi_elec.reshape(-1,1))),  np.array((-np.pi/2.0,0),), self.elec_radius*np.ones((len(self.theta_elec)))*10**2
        J = np.multiply(J, (elec_radius.reshape(J.shape)) ** 2 * np.pi)
        self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        grid_electrode.plot(colorbar='right')
        data = self.electrode_spherical_map

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] >= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 100)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 100)

        spacing_x = (x_lim[1] - x_lim[0]) / 100
        spacing_y = (y_lim[1] - y_lim[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")

        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('x-axis (cm)', fontsize='19')
        plt.ylabel('y-axis (cm)', fontsize='19')
        plt.title('Electrode Pattern Upper Half', fontsize='21')
        plt.tight_layout()
        plt.savefig(fname+"upper.png")
        plt.show()

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] <= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 100)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 100)

        spacing_x = (x_lim[1] - x_lim[0]) / 100
        spacing_y = (y_lim[1] - y_lim[0]) / 100
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")

        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i],1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i],1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('x-axis (cm)', fontsize='19')
        plt.ylabel('y-axis (cm)', fontsize='19')
        plt.title('Electrode Pattern Lower Half', fontsize='21')
        plt.tight_layout()
        plt.savefig(fname+"lower.png")
        plt.show()

@ray.remote
class SparsePlaceRodent(SparsePlaceSpherical):

    def __init__(self, cond_vec, radius_vec, num_elec, elec_radius, elec_spacing,via_size=None, max_l=300, spacing=0.1, custom_grid=False, theta_elec=None, phi_elec=None):

        self.eps = 0.0001*10**(-2)
        self.cond_vec = cond_vec
        self.radius_vec = radius_vec * 10 ** (-2)  ## cm to m
        self.max_l = max_l
        self.spacing =spacing*10**(-2)
        self.r_max = np.max(self.radius_vec)

        self.N_lat = 2 * self.max_l + 2
        self.N_long = self.N_lat

        self.long_phi = np.arange(0, 2 * np.pi, 2 * np.pi / (self.N_long + 1))
        self.lat_theta = np.arange(-np.pi / 2.0, np.pi / 2.0 + self.eps, np.pi / self.N_lat)

        lat_theta_fl, long_phi_fl = np.meshgrid(self.lat_theta, self.long_phi)
        lat_theta_fl, long_phi_fl = lat_theta_fl.flatten(), long_phi_fl.flatten()
        self.electrode_spherical_map_flatten = np.transpose(np.vstack((lat_theta_fl, long_phi_fl)))
        self.electrode_spherical_map = np.zeros((len(self.lat_theta), len(self.long_phi)))
        self.spherical_map_flatten = self.electrode_spherical_map_flatten
        self.num_elec = num_elec
        self.custom_grid = custom_grid

        if self.custom_grid is True:
            self.theta_elec = theta_elec
            self.phi_elec = phi_elec

        self.elec_spacing = elec_spacing * 10**(-2)
        self.elec_radius = elec_radius * 10**(-2)
        self.big_elec_radius = None
        self.big_elec_pos = None
        if via_size is None:
            self.via_size = None
        else:
            self.via_size = via_size*10**(-2)
        self.curr_density_calculated = False

    def forward_model(self, focus_points, cancel_points, print_elec_pattern=False, save=True, save_title_focus="Forward_Focus.npy", save_title_cancel="Forward_Cancel.npy"):

        points = np.vstack((np.array(focus_points), np.array(cancel_points)))
        r = np.unique(points[:,0])

        ## Generate a pseudo-uniform sampling of electrode points
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2,None

        if print_elec_pattern:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=np.ones(len(elec_lst)), via_size=via_size)
            grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
            grid_electrode.plot()

        ## Calculating the Transfer Function
        self.Af = np.zeros((len(focus_points), len(elec_lst)))
        self.Ac = np.zeros((len(cancel_points), len(elec_lst)))

        previous_len_focus = 0
        previous_len_cancel = 0


        for j in range(len(r)):
            _, tau_Jr, _ = self.calc_tauL(r[j])
            tau_Jr= np.hstack((np.array((0,)), tau_Jr))

            idx_focus = np.abs(focus_points[:, 0] - r[j]) < self.eps
            focus_points_subsampled = focus_points[idx_focus]

            idx_cancel = np.abs(cancel_points[:, 0] - r[j]) < self.eps
            cancel_points_subsampled = cancel_points[idx_cancel]

            ## Calculate the Forward Model
            for i in range(len(elec_lst)):

                start_time = time.time()
                elec_pos = np.vstack((elec_lst[i], ground_elec))
                elec_radii = self.elec_radius*np.ones(len(elec_pos))*10**(2)
                inj_curr = np.array((1, -1 * (elec_radius[i] ** 2-via_size[i]**2) / ((self.elec_radius*10**2)**2-via_size[0]**2)))

                # inj_curr = np.array((1,-1))
                self.electrode_sampling(elec_pos, elec_radii, inj_curr, np.array([self.via_size, self.via_size]))
                elec_grid_array = np.flip(self.electrode_spherical_map, axis=0)

                ## Taking the forward spherical harmonic transform
                elec_grid = shp_harm.SHGrid.from_array(elec_grid_array)
                # elec_grid.plot()
                coeff = elec_grid.expand()
                coeff_array = coeff.coeffs

                ## Multipying the transfer function
                curr_density_coeff = np.zeros(coeff_array.shape)
                curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
                curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
                curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

                curr_density = curr_density_coeff.expand(grid='DH')
                curr_density_data = curr_density.data

                ## Focus Points
                for ii in range(len(focus_points_subsampled)):
                    idx1 = np.searchsorted(self.lat_theta, focus_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, focus_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta) - idx1, idx2]
                    self.Af[ii+j*previous_len_focus,i] = current_at_target

                ## Cancel Points
                for ii in range(len(cancel_points_subsampled)):

                    idx1 = np.searchsorted(self.lat_theta, cancel_points_subsampled[ii,1])

                    if idx1 == 0:
                        idx1 = 1
                    idx2 = np.searchsorted(self.long_phi, cancel_points_subsampled[ii,2])
                    if idx2 == len(self.long_phi):
                        idx2 = idx2 - 1

                    current_at_target = curr_density_data[len(self.lat_theta)-idx1, idx2]
                    self.Ac[ii+j*previous_len_cancel,i]= current_at_target

                print("Electrode Remaining:", len(elec_lst)-i-1, "Time Taken:", time.time()-start_time)

            previous_len_focus = previous_len_focus+len(focus_points_subsampled)
            previous_len_cancel = previous_len_cancel+len(cancel_points_subsampled)
        if save:
            np.save(save_title_focus, self.Af)
            np.save(save_title_cancel, self.Ac)
        return self.Af, self.Ac

    def get_Af_and_Ac(self):
        return self.Af, self.Ac

    def calc_voltage(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        tau_V, _, _ = self.calc_tauL(r)
        tau_V = np.hstack((np.array((0,)), tau_V))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_V)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_V)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.volatge_calculated = True
        self.voltage_at_target = curr_density.data*10
        return self.voltage_at_target

    def calc_curr_density(self, r, J=None):
        ## Electrode Density
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        if J is None:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=self.J_sparseplace_vanilla, via_size=via_size)
        else:
            self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J, via_size=via_size)

        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        # grid_electrode.plot(colorbar='right')

        ## Current Density
        _, tau_Jr, _ = self.calc_tauL(r)
        tau_Jr = np.hstack((np.array((0,)), tau_Jr))

        coeff = grid_electrode.expand()
        coeff_array = coeff.coeffs

        ## Multipying the transfer function
        curr_density_coeff = np.zeros(coeff_array.shape)
        curr_density_coeff[0, :, :] = np.transpose(np.transpose(coeff_array[0]) * tau_Jr)
        curr_density_coeff[1, :, :] = np.transpose(np.transpose(coeff_array[1]) * tau_Jr)
        curr_density_coeff = shp_harm.SHCoeffs.from_array(curr_density_coeff)

        curr_density = curr_density_coeff.expand(grid='DH')
        # curr_density.plot(colorbar='right')
        self.curr_density_calculated = True
        self.curr_density_at_target = curr_density.data
        return self.curr_density_at_target

    def plot_elec_pttrn(self, J, x_lim=None, y_lim=None, fname=None):

        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None
        J = np.multiply(J, ((elec_radius.reshape(J.shape)) ** 2-(via_size.reshape(J.shape))**2) * np.pi)

        self.electrode_sampling(elec_pos=elec_lst, elec_radii=elec_radius, injected_curr=J, via_size=via_size)
        grid_electrode = shp_harm.SHGrid.from_array(np.flip(self.electrode_spherical_map, axis=0), grid='DH')
        grid_electrode.plot(colorbar='right')
        data = self.electrode_spherical_map

        long_fl, lat_fl = np.meshgrid(self.long_phi, self.lat_theta)
        long_fl, lat_fl = long_fl.flatten(), lat_fl.flatten()
        spherical_map_flatten = np.transpose(np.vstack((long_fl, lat_fl)))

        ### Plotting the first half
        idx = spherical_map_flatten[:, 1] >= 0
        x, y = self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.cos(spherical_map_flatten[idx, 0]), self.r_max * 10**2*np.cos(spherical_map_flatten[idx, 1]) * np.sin(spherical_map_flatten[idx, 0])
        data_flatten = data.flatten()
        data_flatten = data_flatten[idx]
        if x_lim is None:
            x_lim = np.array((np.min(x), np.max(x)))
        if y_lim is None:
            y_lim = np.array((np.min(y), np.max(y)))
        x_discretize = np.arange(x_lim[0], x_lim[1] + self.eps, (x_lim[1] - x_lim[0]) / 130)
        y_discretize = np.arange(y_lim[0], y_lim[1] + self.eps, (y_lim[1] - y_lim[0]) / 130)

        spacing_x = (x_lim[1] - x_lim[0]) / 130
        spacing_y = (y_lim[1] - y_lim[0]) / 130
        data_projected = np.zeros((len(x_discretize), len(y_discretize)))
        for i in range(len(x_discretize)):
            for j in range(len(y_discretize)):
                data_projected[i, j] = np.mean(data_flatten[np.where(np.square(x - x_discretize[i]) + np.square(y - y_discretize[j]) <= spacing_x ** 2 + spacing_y ** 2)])
        data_projected = np.transpose(data_projected)
        data_projected = np.flip(data_projected, axis=0)
        sns.heatmap(data_projected, cmap="jet")

        labels_x = np.linspace(x_lim[0], x_lim[1], 11)
        labels_x = [str(round(labels_x[i]*10,1))[0:4] for i in range(len(labels_x))]
        labels_y = np.linspace(y_lim[0], y_lim[1], 11)
        labels_y = [str(round(labels_y[i]*10,1))[0:4] for i in range(len(labels_y))]
        labels_y = np.flip(np.array(labels_y))
        x_ticks = np.linspace(0,len(x_discretize),len(labels_x))
        y_ticks = np.linspace(0, len(y_discretize), len(labels_y))
        plt.xticks(x_ticks, labels_x, fontsize='15')
        plt.yticks(y_ticks, labels_y, fontsize='15')
        plt.xlabel('x-axis (mm)', fontsize='19')
        plt.ylabel('y-axis (mm)', fontsize='19')
        plt.title('Electrode Pattern', fontsize='21')
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def minimize_linfty(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm_inf(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def minimize_l2(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def minimize_l1(self, Jdes, Af=None):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0]#, cvx.atoms.norm1(I)<=beta+1]
        obj = cvx.Minimize(cvx.atoms.norm(I))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        return I.value

    def SparsePlaceVanilla(self, Jdes, Jsafety, Af=None, Ac=None, beta=0.001):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None

        elec_radius = elec_radius.reshape(1, -1)
        area_elec = (elec_radius ** 2 - via_size ** 2) * np.pi
        I = cvx.Variable(Af.shape[1])
        if hasattr(Jsafety, "__len__"):
            constraints = [Af @ I == Jdes, cvx.sum(area_elec @ I) == 0, cvx.atoms.abs(I) <= Jsafety, cvx.atoms.norm1(I) <= beta]
        else:
            constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.norm_inf(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        obj = cvx.Minimize(cvx.atoms.norm(Ac@I)) #+beta*cvx.atoms.norm1(I)
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def SparsePlaceHinge(self, Jdes, Jsafety, Jtol, Af=None, Ac=None, beta=0.001):
        if self.custom_grid is False:
            if self.big_elec_radius is None:
                elec_lst, ground_elec, elec_radius, via_size = self.rodent_patch(elec_spacing=self.elec_spacing*10**(2), elec_radius=self.elec_radius*10**2, num_elec=self.num_elec, via_size=self.via_size*10**2)
            else:
                elec_lst, ground_elec, elec_radius, via_size = self.NHP_patch_big_elec(elec_spacing=self.elec_spacing * 10 ** (2), elec_radius=self.elec_radius * 10 ** 2, num_elec=self.num_elec, elec_patch_angle=self.elec_patch_angle, via_size=self.via_size * 10 ** 2, big_elec_radius=self.big_elec_radius, big_elec_pos=self.big_elec_pos)
        else:
            if self.via_size is not None:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, self.via_size * np.ones((len(self.theta_elec))) * 10 ** 2
            else:
                elec_lst, ground_elec, elec_radius, via_size = np.hstack((self.theta_elec.reshape(-1, 1), self.phi_elec.reshape(-1, 1))), np.array((-np.pi / 2.0, 0), ), self.elec_radius * np.ones((len(self.theta_elec))) * 10 ** 2, None


        elec_radius = elec_radius.reshape(1,-1)
        area_elec = (elec_radius**2-via_size**2)*np.pi
        I = cvx.Variable(Af.shape[1])
        if hasattr(Jsafety, "__len__"):
            constraints = [Af @ I == Jdes, cvx.sum(area_elec@I) == 0, cvx.atoms.abs(I)<=Jsafety, cvx.atoms.norm1(I)<=beta]
        else:
            constraints = [Af @ I == Jdes, cvx.sum(area_elec @ I) == 0, cvx.atoms.norm_inf(I) <= Jsafety, cvx.atoms.norm1(I) <= beta]
        obj = cvx.Minimize(cvx.sum(cvx.atoms.maximum(0, Ac@I-Jtol*np.ones((Ac.shape[0]))))+cvx.sum(cvx.atoms.maximum(0, -Ac@I-Jtol*np.ones((Ac.shape[0])))))
        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=True, solver=cvx.ECOS, max_iters=300)  # Returns the optimal value.
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("constraint", np.matmul(Af.reshape(1,-1), I.value))
        print('constraint:', np.sum(np.multiply(I.value, area_elec)))
        if prob.status == 'infeasible':
            return None
        else:
            return I.value

    def run_diffused(self, per, Jdes, focus_points, Af=None, print_elec_pattern=False,plot=True, fname_density=None, fname_area=None, fname_elec=None, density_thresh=None):
        print(">>>>>>>>>Running Hinge Loss SparsePlace<<<<<<<<<<<")

        self.J_l2 = self.minimize_l2(Jdes=Jdes, Af=Af)
        self.J_linfty = self.minimize_linfty(Jdes=Jdes, Af=Af)
        self.J_l1 = self.minimize_l1(Jdes=Jdes, Af=Af)


        # curr_density = self.calc_curr_density(r=np.min(focus_points[:, 0]), J=self.J_sparseplace_hinge)
        # print("Error Factor:", Jdes / np.max(curr_density))
        #self.J_sparseplace_hinge = Jdes / np.max(curr_density) * self.J_sparseplace_hinge
        if print_elec_pattern:
            self.plot_elec_pttrn(self.J_l2, fname=fname_elec+"l2.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
            self.plot_elec_pttrn(self.J_l1, fname=fname_elec + "l1.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
            self.plot_elec_pttrn(self.J_linfty, fname=fname_elec + "linfty.png", x_lim=np.array((-1.5, 1.5)), y_lim=np.array((-1, 1)))
        if plot is True:
            self.plot_curr_density(r=np.min(focus_points[:,0]), J=self.J_l2, fname=fname_density+"_l2.png", x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), abs=False)
            self.plot_curr_density(r=np.min(focus_points[:, 0]), J=self.J_l1, fname=fname_density+"_l2.png", x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), abs=False)
            self.plot_curr_density(r=np.min(focus_points[:, 0]), J=self.J_linfty, fname=fname_density+"_l2.png", x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), abs=False)
        if density_thresh is None:
            area_l2 = self.area_above_x(r=np.min(focus_points[:,0]), per=per, plot=plot, J=self.J_l2,x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area+"_l2.png")
            area_l2 = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_l2, x_limit=np.array((-2, 2)), y_limit=np.array((-2, 2)), fname=fname_area + "_l2.png")
        else:
            area = self.area_above_x(r=np.min(focus_points[:, 0]), per=per, plot=plot, J=self.J_sparseplace_hinge, x_limit=np.array((-2,2)), y_limit=np.array((-2,2)), fname=fname_area, density_thresh=density_thresh)

        return self.J_l2,self.J_linfty,self.J_l1, area



