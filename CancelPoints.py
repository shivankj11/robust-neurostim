import numpy as np
import matplotlib.pyplot as plt

class CancelPointsSpherical:

    def __init__(self, spacing):
        self.spacing = spacing*10**(-2)

    def cancel_points_sampling(self, r):

        r = r*10**(-2)
        theta_loc = np.arange(np.pi / 2.0, -np.pi / 2.0, -1 * self.spacing / r)
        theta_loc = theta_loc[1:]
        self.spherical_cancel_points = np.array((np.pi / 2.0, 0)).reshape(1, 2)

        for i in range(len(theta_loc)):
            r_ring = r * np.cos(theta_loc[i])
            num_elec = np.floor(2 * np.pi * r_ring /self.spacing)
            phi_loc = np.arange(0, np.pi * 2, 2 * np.pi / num_elec)
            temp = np.transpose(np.array((theta_loc[i] * np.ones(len(phi_loc)), phi_loc)))
            self.spherical_cancel_points = np.vstack((self.spherical_cancel_points, temp))

    def uniform_ring_sample(self, centre, offset, thickness):

        self.cancel_points_sampling(centre[0])
        lat_diff = self.spherical_cancel_points[:, 0] - centre[1]
        long_diff = self.spherical_cancel_points[:, 1] - centre[2]

        dist = np.sin(lat_diff / 2.0) ** 2 + np.cos(self.spherical_cancel_points[:, 0]) * np.cos(centre[1] * np.ones(len(self.spherical_cancel_points[:, 0]))) * np.sin(long_diff / 2.0) ** 2
        dist = centre[0] * 2 * np.arcsin(np.sqrt(dist))
        idx = dist<=offset+thickness
        cand_points = self.spherical_cancel_points[idx]
        idx = dist[idx]>=offset
        cand_points = cand_points[idx]
        self.cand_points = np.hstack((centre[0]*np.ones((len(cand_points),1)),cand_points))
        return self.cand_points

    def end_point_sample(self, theta_start, theta_end, phi_start, phi_end, r, spacing=None):

        r = r*10**(-2)
        if spacing is None:
            spacing = self.spacing
        else:
            spacing = spacing*10**(-2)

        theta_loc = np.arange(theta_start, theta_end, -1 * spacing / r)
        if theta_start == np.pi/2.0:
            theta_loc = theta_loc[1:]
            self.spherical_cancel_points = np.array((np.pi / 2.0, 0)).reshape(1, 2)
            for i in range(len(theta_loc)):
                r_ring = r * np.cos(theta_loc[i])
                num_elec = np.floor(np.abs(phi_end-phi_start)*r_ring / spacing)
                phi_loc = np.arange(phi_start, phi_end, np.abs(phi_end-phi_start) / num_elec)
                temp = np.transpose(np.array((theta_loc[i] * np.ones(len(phi_loc)), phi_loc)))
                self.spherical_cancel_points = np.vstack((self.spherical_cancel_points, temp))
        else:
            for i in range(len(theta_loc)):
                r_ring = r * np.cos(theta_loc[i])
                num_elec = np.floor(np.abs(phi_end-phi_start)*r_ring / spacing)
                phi_loc = np.arange(phi_start, phi_end, np.abs(phi_end-phi_start) / num_elec)
                temp = np.transpose(np.array((theta_loc[i] * np.ones(len(phi_loc)), phi_loc)))
                if i==0:
                    self.spherical_cancel_points = temp
                else:
                    self.spherical_cancel_points = np.vstack((self.spherical_cancel_points, temp))

        self.cand_points = np.hstack((r * 10**(2)*np.ones((len(self.spherical_cancel_points), 1)), self.spherical_cancel_points))
        return self.cand_points