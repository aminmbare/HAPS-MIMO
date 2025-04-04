import numpy as np
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from shapely.geometry import Point, Polygon
import os
import random
class MU_MIMO_CapacityCalculator:
    def __init__(self, small_scale_fading, path_loss_calculator, noise_power_density, bandwidth):
        """
        Initializes the MU-MIMO capacity calculator with the necessary components.
        
        Parameters:
        small_scale_fading (SmallScaleFading): Instance to generate small-scale fading matrices.
        path_loss_calculator (PathLossCalculator): Instance for path loss calculations.
        interference_calculator (DynamicClusterInterferenceCalculator): Instance for calculating inter-cluster interference.
        noise_power_density (float): Noise power density (W/Hz).
        bandwidth (float): System bandwidth in Hz.
        """
        self.small_scale_fading = small_scale_fading
        self.path_loss_calculator = path_loss_calculator
        self.noise_power_density = noise_power_density
        self.bandwidth = bandwidth
        self.P_tx = 41
        self.G_tx = 28
        self.G_rx = 0 
        
 
        
    def RSPR_cluster_paramters(self, cluster, haps_coordinates, sinr_map = False):   
        
    
        N_sim = cluster['N_sim']
        SBAND_data = cluster['SBAND_data']
        area_type = cluster['area_type']
        if area_type == 'dense_urban':
            scenario_data = SBAND_data.dense_urban
        elif area_type == 'urban':
            scenario_data = SBAND_data.urban
        else:
            scenario_data = SBAND_data.suburban_rural
            
        los_probability = scenario_data.los_probability
        path_loss_parameters = scenario_data.path_loss
        #if not sinr_map : 
        #    polygon = cluster['boundary']
        #    #print(area_coordinates)
        #    min_x, min_y, max_x, max_y = polygon.bounds
        #    
        #    while True:
        #        area_coordinates = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        #        if polygon.contains(area_coordinates):
        #            area_coordinates = [area_coordinates.x, area_coordinates.y, 0]
        #            break  # Valid initial point found
        #else : 
        area_coordinates = cluster['area_coordinates']
        elevation_angle_deg = cluster['elevation_angle_deg']
        total_path_loss = 0
        indoor_percentage = cluster['indoor_percentage']
        traditional_building_percentage = cluster['traditional_building_percentage']
        
        total_small_scale_fading = np.zeros(16, dtype=complex)
        #print(los_probability)
        for _ in range(N_sim) : 
            if np.random.rand() < indoor_percentage:
                los = False
            else : 
                if np.random.rand() < los_probability / 100:
                    los = True
                else:
                    los = False
            
            # calculate path loss 
            total_path_loss+= self.path_loss_calculator.calculate_path_loss(area_coordinates, haps_coordinates,elevation_angle_deg, los, path_loss_parameters, indoor_percentage, traditional_building_percentage)  
            total_small_scale_fading += self.small_scale_fading.calculate_total_H(haps_coordinates, area_coordinates, los)
            # calculate interference power
        small_scale_fading = total_small_scale_fading / N_sim
        path_loss = total_path_loss / N_sim
        return small_scale_fading, path_loss
    
    def generate_capacities(self, clusters_list, haps_coordinates, Nr, Nt, sigma_squared = 1e-2,plot = False, map_boundary_points = None):
        """
        Generate the Signal-to-Interference-plus-Noise Ratio (SINR) for a given cluster.
        
        Parameters:
        cluster (dict): Dictionary containing cluster information.
        haps_coordinates (List): HAPS coordinates [x, y, z].
        elevation_angle_deg (float): Elevation angle in degrees.
        
        Returns:
        float: SINR in dB.
        """
        # Calculate the small-scale fading and path loss for the cluster
        H_matrix = np.zeros((Nr, Nt), dtype=complex)
        #H_matrix_test = np.zeros((Nr, Nt), dtype=complex)
        #self.small_scale_fading.Nr = Nr
        for i,cluster in enumerate(clusters_list):
            #print(cluster)
            #print(cluster["area_coordinates"])
            small_scale_fading, path_loss = self.RSPR_cluster_paramters(cluster, haps_coordinates)
            
            
            H_matrix[i] = np.sqrt(10**((self.G_tx-path_loss)/10)) * small_scale_fading
            #H_matrix_test[i] = small_scale_fading
            #print("path loss", path_loss)
            #
            # print("small scale fading", small_scale_fading)
        #print(H_matrix, H_matrix.shape)
        #print(H_matrix @ H_matrix.conj().T)
        
        W_mmse = np.linalg.inv(  H_matrix.conj().T @ H_matrix+ sigma_squared * np.eye(H_matrix.shape[1])) @ H_matrix.conj().T
        #W_mmse_test =  np.linalg.inv(  H_matrix_test.conj().T @ H_matrix_test+ sigma_squared * np.eye(H_matrix_test.shape[1])) @ H_matrix_test.conj().T
        #print("W_mmse",W_mmse)
        #print("W_mmse_test",W_mmse_test)
        #print("H matrix", H_matrix)
        #print("WMMSE",W_mmse)
        W_mmse_copy = W_mmse.copy()
        #print("WWMSE",W_mmse.shape)
        for i in range(W_mmse.shape[1]):
            #print("norm",np.linalg.norm(W_mmse[:, i]))
            W_mmse[:, i] /= np.linalg.norm(W_mmse[:, i])
        #interference_power 
        
        capacities = []
        for cluster_id in range(H_matrix.shape[0]):
            h_i = H_matrix[cluster_id]
            w_i = W_mmse[:, cluster_id]  # Beamforming vector for user i
            #print("Gain",np.abs(np.dot(h_i, w_i))**2)
            # Signal power for user i
            signal_power = np.abs(np.dot(h_i, w_i))**2 * 10**((self.P_tx)/10)
            #print(8 / (signal_power/10**(self.noise_power_density/10)))
            # Interference power from other cluster beams
            interference_power = sum(
                np.abs(np.dot(h_i, W_mmse[:, j]))**2 * 10**((self.P_tx)/10) for j in range(H_matrix.shape[0]) if j != cluster_id
            )
            #print("power", signal_power)
            #print("inteferece power", interference_power)   
            noise_power = sigma_squared * self.bandwidth  # Total noise power
            SINR_i = signal_power / (interference_power + 10**(self.noise_power_density/10))
            SNR_i = signal_power / 10**(self.noise_power_density/10)
     
            # Calculate capacity for user i
            capacity_i = self.bandwidth * np.log2(1 + SINR_i)
            capacities.append(capacity_i)
        if plot : 
            #self.visualize_beamforming_pattern_3D(clusters_list, haps_coordinates, W_mmse, 4, 2)
            #self.visualize_beamforming_with_corrected_directions(W_mmse, 4, 2, clusters_list, haps_coordinates, [cluster['area_coordinates'] for cluster in clusters_list])
            self.visualize_beamforming_patterns_superimposed_3D(W_mmse, 4, 4, cluster_list=clusters_list)
                # Visualization of beamforming directions
            #self.visualize_beamforming_pattern_planar(W_mmse, 4, 4,cluster_list= clusters_list)
            #self.compute_interference_2d_plane(W_mmse, haps_coordinates, clusters_list)
            
            #self.compute_interference_with_bounds(W_mmse, haps_coordinates, clusters_list)
        
        return capacities

    def set_SINR_map(self,map_bounds, grid_size=150): 
        """
        Set the map boundaries for SINR visualization.
        
        Parameters:
        map_bounds (list): List containing [min_x, min_y, max_x, max_y] for the map.
        """

        min_x, min_y, max_x, max_y = map_bounds  # Extract map bounds

        # Define the 2D grid within the bounds
        x_range = np.linspace(min_x, max_x, grid_size)
        y_range = np.linspace(min_y, max_y, grid_size)
        self.grid_size = grid_size 
        self.X, self.Y = np.meshgrid(x_range, y_range)
        self.SINR_map = np.zeros_like(self.X)
    def compute_interference_with_bounds(self, W_mmse, haps_coordinates, clusters_list):
        """
        Compute and visualize the interference level over a 2D grid, considering area boundaries and map bounds.

        Parameters:
        W_mmse (array): MMSE beamforming weights (N_antennas x N_clusters).
        haps_coordinates (list): HAPS coordinates [x, y, z].
        clusters_list (list): List of cluster dictionaries with 'area_name' and 'boundary' (Shapely Polygon).
        map_bounds (list): List containing [min_x, min_y, max_x, max_y] for the map.
        grid_size (int): Number of grid points per axis.
        """

        print("computing maps interference")
        N_clusters = len(clusters_list)

        # Loop through each grid point
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid_x = self.X[i, j]
                grid_y = self.Y[i, j]
                tile_point = Point(grid_x, grid_y)

                assigned_beam = None
                #print("tile point", i , j)
                # Check which area the grid point belongs to
                for cluster_id, cluster in enumerate(clusters_list):
                    boundary = cluster['boundary']  # Shapely Polygon
                    if boundary.contains(tile_point):
                        assigned_beam = cluster_id
                        #print("assigned beam", assigned_beam)
                        break

                # Skip if no beam covers this grid point
                if assigned_beam is None:
                    #print("assigned beam", assigned_beam)
                    continue

                # Compute path loss for the grid point
                #distance_to_haps = np.sqrt((grid_x - haps_coordinates[0])**2 +
                #                        (grid_y - haps_coordinates[1])**2 +
                #                        haps_coordinates[2]**2)
                cluster["N_sim"] = 100
                cluster["area_coordinates"] = [grid_x, grid_y, 0]
                small_scale_fading, path_loss = self.RSPR_cluster_paramters(cluster, haps_coordinates, sinr_map = True)
                h_tile = np.sqrt(10**((self.G_tx - path_loss) / 10)) * small_scale_fading
                signal_power = np.abs(np.dot(h_tile, W_mmse[:,assigned_beam]))**2 * 10**((self.P_tx)/10)
                #h_tile = np.sqrt(10**((self.G_tx - path_loss) / 10)) * small_scale_fading

                # Compute interference power (sum contributions from all beams except the assigned one)
                interference_power = sum(
                    np.abs(np.dot(h_tile, W_mmse[:, k]))**2 * 10**(self.P_tx / 10)
                    for k in range(N_clusters) if k != assigned_beam
                )
                SINR_i = signal_power / (interference_power + 10**(self.noise_power_density/10))
                self.SINR_map[i, j] +=  SINR_i

        
        
        
        
    def compute_SINR_map(self, clusters_list: List[dict], area_selection_counter , path):
        # Plot the interference heatmap
        np.save(os.path.join(path, "SINR_map.npy"), self.SINR_map)
        np.save(os.path.join(path, "area_selectiion.npy"), area_selection_counter)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid_x = self.X[i, j]
                grid_y = self.Y[i, j]
                tile_point = Point(grid_x, grid_y)

                assigned_beam = None
                #print("tile point", i , j)
                # Check which area the grid point belongs to
                for cluster_id, cluster in enumerate(clusters_list):
                    boundary = cluster['boundaries']  # Shapely Polygon
                    if boundary.contains(tile_point):
                        assigned_beam = cluster_id
                 #       print("assigned beam", assigned_beam)
                        self.SINR_map[i,j] = self.SINR_map[i,j] / area_selection_counter[assigned_beam]
                        break

                # Skip if no beam covers this grid point
                if assigned_beam is None:
                    #print("assigned beam", assigned_beam)
                    continue
        plt.figure(figsize=(10, 8))
        plt.contourf(self.X, self.Y, 10 * np.log10(self.SINR_map ), levels=50, cmap='viridis')
        plt.colorbar(label='Interference Power (dB)')
        plt.title("Interference Heatmap Within Map and Area Boundaries")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        # Overlay area boundaries
        for cluster in clusters_list:
            boundary = cluster['boundaries']
            area_x, area_y = boundary.exterior.xy
            plt.plot(area_x, area_y, linewidth=2, label=cluster['area_name'])

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(path, "interference_heatmap.png"))
        plt.close()
        #1np.save(path, self.SINR_map)

        
    @staticmethod
    def visualize_beamforming_pattern_planar(W_mmse, N_x, N_y,  cluster_list ,N_phi=360,N_theta=90):
        """
        Visualize beamforming gain over azimuth and elevation directions for a planar array.

        Parameters:
        W_mmse (array): Beamforming weights for a specific cluster.
        N_x (int): Number of antennas along x-axis.
        N_y (int): Number of antennas along y-axis.
        N_phi (int): Number of azimuth angles.
        N_theta (int): Number of elevation angles.
        """
        N_antennas, N_clusters = W_mmse.shape
        assert N_antennas == N_x * N_y, f"Mismatch: Antennas {N_antennas} != {N_x * N_y}"

        # Define azimuth angles (only plot azimuth for simplicity)
        phi = np.linspace(0, 2 * np.pi, N_phi)- np.pi  # Azimuth angles [0, 360 degrees]
        theta = np.pi  # Fixed elevation angle (45 degrees, for simplicity)

        # Steering vector for a planar array
        def steering_vector_planar_fixed_elevation(phi, N_x, N_y, theta=np.pi / 6):
            """Generate the flattened steering vector for a planar array at fixed elevation."""
            n_x = np.arange(N_x)[:, None]  # Antenna indices along x
            n_y = np.arange(N_y)[None, :]  # Antenna indices along y
            # Compute phase shifts with fixed elevation theta
            phase_x = np.exp(1j * np.pi * n_x * np.sin(theta) * np.cos(phi))
            phase_y = np.exp(1j * np.pi * n_y * np.sin(theta) * np.sin(phi))
            return (phase_x * phase_y).flatten()  # Flatten to 1D

        # Initialize the polar plot
        for i in range(2,9):
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Loop over each cluster (beam) and plot its gain
            for cluster_id in range(N_clusters):
                area_names = cluster_list[cluster_id]['area_name']
                W_cluster = W_mmse[:, cluster_id]  # Weights for this cluster
                gain = np.zeros(N_phi)  # Initialize gain over azimuth

                # Compute beamforming gain for all azimuth angles
                for j, p in enumerate(phi):
                    a_phi = steering_vector_planar_fixed_elevation(p, N_x, N_y,theta/i)  # Steering vector at azimuth p
                    gain[j] = np.abs(np.dot(W_cluster.conj().T, a_phi))**2  # Dot product gain

                # Normalize gain for this cluster
                gain = gain / np.max(gain)

                # Plot the beamforming pattern for this cluster
                ax.plot(phi, gain, label=f"{area_names}")

            # Aesthetic adjustments
            ax.set_title("Superimposed Beamforming Patterns for All Clusters")
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
            #plt.savefig(f"results/plot_beam/2d/beamforming_pattern_planar_{int(theta/i)}.png")
            #plt.close(fig)
            plt.show()
    
    @staticmethod
    def visualize_beamforming_patterns_superimposed_3D(W_mmse, N_x, N_y, cluster_list, N_phi=360, N_theta=90):
        """
        Visualize superimposed beamforming gains over azimuth and elevation directions for a planar array in 3D.

        Parameters:
        W_mmse (array): Beamforming weights (shape: N_antennas x N_clusters).
        N_x (int): Number of antennas along x-axis.
        N_y (int): Number of antennas along y-axis.
        cluster_list (list): List of clusters, each with 'area_name'.
        N_phi (int): Number of azimuth angles.
        N_theta (int): Number of elevation angles.
        """
        N_antennas, N_clusters = W_mmse.shape
        assert N_antennas == N_x * N_y, f"Mismatch: Antennas {N_antennas} != {N_x * N_y}"

        # Define azimuth and elevation angles
        phi = np.linspace(-np.pi, np.pi, N_phi)  # Azimuth angles [-180, 180 degrees]
        theta = np.linspace(0, np.pi / 2, N_theta)  # Elevation angles [0, 90 degrees]

        # Steering vector for a planar array
        def steering_vector_planar(theta, phi, N_x, N_y):
            """Generate the flattened steering vector for a planar array at (theta, phi)."""
            n_x = np.arange(N_x)[:, None]  # Antenna indices along x
            n_y = np.arange(N_y)[None, :]  # Antenna indices along y
            # Compute phase shifts
            phase_x = np.exp(1j * np.pi * n_x * np.sin(theta) * np.cos(phi))
            phase_y = np.exp(1j * np.pi * n_y * np.sin(theta) * np.sin(phi))
            return (phase_x * phase_y).flatten()  # Flatten to 1D

        # Prepare 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']  # Distinct colors for clusters

        # Loop over each cluster (beam)
        for cluster_id in range(N_clusters):
            area_names = cluster_list[cluster_id]['area_name']
            W_cluster = W_mmse[:, cluster_id]  # Weights for this cluster

            # Compute 3D beamforming gain
            gain = np.zeros((N_theta, N_phi))  # Grid for gain (elevation × azimuth)
            for i, t in enumerate(theta):
                for j, p in enumerate(phi):
                    a_theta_phi = steering_vector_planar(t, p, N_x, N_y)  # Steering vector
                    gain[i, j] = np.abs(np.dot(W_cluster.conj().T, a_theta_phi))**2  # Dot product gain

            # Normalize gain for this cluster
            gain = gain 
            
            # Convert polar to Cartesian for 3D plotting
            azimuth, elevation = np.meshgrid(phi, theta)
            x = gain * np.sin(elevation) * np.cos(azimuth)
            y = gain * np.sin(elevation) * np.sin(azimuth)
            z = gain * np.cos(elevation)
            #z = 10*np.log10(z)
            # Plot surface for the current cluster
            ax.plot_surface(x, y, z, color=colors[cluster_id % len(colors)], alpha=0.6, label=area_names, edgecolor='none')

        # Add labels and legend
        ax.set_title("Superimposed 3D Beamforming Patterns")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Gain (dBi)")
        #ax.legend([cluster['area_name'] for cluster in cluster_list], loc='upper right', bbox_to_anchor=(1.4, 1.0)) 
        plt.savefig("results/plot_beam/3d/superimposed_beamforming_patterns.png")
        plt.show()
    @staticmethod
    def visualize_beamforming_with_corrected_directions(W_mmse, N_x, N_y, cluster_list, haps_position, cluster_positions, N_phi=360, N_theta=90):
        """
        Visualize beamforming patterns from HAPS toward cluster positions in 3D, with corrected directions.

        Parameters:
        W_mmse (array): Beamforming weights (shape: N_antennas x N_clusters).
        N_x (int): Number of antennas along x-axis.
        N_y (int): Number of antennas along y-axis.
        cluster_list (list): List of clusters, each with 'area_name'.
        haps_position (tuple): Position of HAPS (x, y, z).
        cluster_positions (list): List of cluster positions [(x_c, y_c, z_c), ...].
        N_phi (int): Number of azimuth angles.
        N_theta (int): Number of elevation angles.
        """
        N_antennas, N_clusters = W_mmse.shape
        assert N_antennas == N_x * N_y, f"Mismatch: Antennas {N_antennas} != {N_x * N_y}"

        # HAPS position
        haps_x, haps_y, haps_z = haps_position

        # Define azimuth and elevation angles
        phi = np.linspace(-np.pi, np.pi, N_phi)  # Azimuth angles [-180, 180 degrees]
        theta = np.linspace(0, -np.pi/2 , N_theta)  # Elevation angles [0, 90 degrees]

        # Steering vector for a planar array
        def steering_vector_planar(theta, phi, N_x, N_y):
            """Generate the flattened steering vector for a planar array at (theta, phi)."""
            n_x = np.arange(N_x)[:, None]  # Antenna indices along x
            n_y = np.arange(N_y)[None, :]  # Antenna indices along y
            # Compute phase shifts
            phase_x = np.exp(1j * np.pi * n_x * np.sin(theta) * np.cos(phi))
            phase_y = np.exp(1j * np.pi * n_y * np.sin(theta) * np.sin(phi))
            return (phase_x * phase_y).flatten()  # Flatten to 1D

        # Prepare 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']  # Distinct colors for clusters

        # Loop over each cluster
        for cluster_id in range(N_clusters):
            area_names = cluster_list[cluster_id]['area_name']
            W_cluster = W_mmse[:, cluster_id]  # Weights for this cluster
            cluster_x, cluster_y, cluster_z = cluster_positions[cluster_id]

            # Compute azimuth and elevation angles toward the cluster
            relative_x = cluster_x - haps_x
            relative_y = cluster_y - haps_y
            relative_z = cluster_z - haps_z

            cluster_phi = np.arctan2(relative_y, relative_x)  # Azimuth angle
            cluster_theta = np.arctan2(relative_z, np.sqrt(relative_x**2 + relative_y**2))  # Elevation angle

            # Compute 3D beamforming gain
            gain = np.zeros((N_theta, N_phi))  # Grid for gain (elevation × azimuth)
            for i, t in enumerate(theta):  # Elevation sweep
                for j, p in enumerate(phi):  # Azimuth sweep
                    a_theta_phi = steering_vector_planar(t + cluster_theta, p + cluster_phi, N_x, N_y)
                    gain[i, j] = np.abs(np.dot(W_cluster.conj().T, a_theta_phi))**2

            # Normalize gain for this cluster
            gain = gain / np.max(gain)
            gain*= 2*1e4
            # Convert polar to Cartesian for 3D plotting
            azimuth, elevation = np.meshgrid(phi, theta)
            x = haps_x + gain * np.sin(elevation) * np.cos(azimuth)
            y = haps_y + gain * np.sin(elevation) * np.sin(azimuth)
            z = haps_z + gain * np.cos(elevation)

            # Plot surface for the current cluster
            ax.plot_surface(x, y, z, color=colors[cluster_id % len(colors)], alpha=0.6, edgecolor='none', label=area_names)

        # Plot cluster positions
        for i, cluster_position in enumerate(cluster_positions):
            cluster_x, cluster_y, cluster_z = cluster_position
            ax.scatter(cluster_x, cluster_y, cluster_z, color=colors[i % len(colors)], s=100, label=f"{cluster_list[i]['area_name']} (Cluster Pos)")

        # Add HAPS marker
        ax.scatter(haps_x, haps_y, haps_z, color="black", s=150, marker="o", label="HAPS Position")

        # Add labels and legend
        ax.set_title("Beamforming Patterns Toward Clusters")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        #ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
        plt.savefig("results/plot_beam/3d/targeted_beamforming_patterns_corrected.png")
        plt.show()

    
        
    @staticmethod
    def visualize_beam_lobes(clusters_list, haps_coordinates, W_mmse, Nx, Ny, dx=0.5, dy=0.5, wavelength=0.1):
        """
        Visualize 3D beam patterns with main lobe matching the cluster color.

        Parameters:
        clusters_list (list): List of clusters with their [x, y, z] positions.
        haps_coordinates (list): Coordinates of the HAPS [x, y, z].
        W_mmse (np.ndarray): MMSE beamforming weights (Nx * Ny x number of clusters).
        Nx (int): Number of antennas along the x-axis.
        Ny (int): Number of antennas along the y-axis.
        dx (float): Spacing between antennas along the x-axis (in multiples of wavelength).
        dy (float): Spacing between antennas along the y-axis (in multiples of wavelength).
        wavelength (float): Wavelength of the signal.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot HAPS position
        haps_x, haps_y, haps_z = haps_coordinates
        ax.scatter(haps_x, haps_y, haps_z, color='red', s=100, label='HAPS', edgecolor='black')

        # Compute the wave number
        k = 2 * np.pi / wavelength

        # Define a list of colors for clusters
        # Example colors list
        cluster_colors = [
            "blue", "orange", "green", "red", 
            "purple", "brown", "pink", "gray"
        ]
      # Flatten to 1D
     
        # Loop over each cluster
        for cluster_id, cluster in enumerate(clusters_list):
            cluster_x, cluster_y, cluster_z = cluster['area_coordinates']
            
            # Plot cluster position
            cluster_color = cluster_colors[cluster_id % len(cluster_colors)]  # Assign color cyclically
            ax.scatter(cluster_x, cluster_y, cluster_z, s=80, color=cluster_color, label=f'{cluster["area_name"]}')

            # Compute beam direction (vector from HAPS to cluster)
            direction = np.array([cluster_x - haps_x, cluster_y - haps_y, cluster_z - haps_z])
            direction = direction / np.linalg.norm(direction)  # Normalize direction vector

            # Compute azimuth and elevation angles
            azimuth_cluster = np.arctan2(direction[1], direction[0])- np.pi  # Azimuth (phi)
            elevation_cluster = np.arcsin(direction[2]) - np.pi/2 # Elevation (theta)

            # Grid for azimuth and elevation angles (centered around cluster direction)
            azimuths = np.linspace(azimuth_cluster - np.pi , azimuth_cluster + np.pi , 150)  # Azimuth angles
            elevations = np.linspace(elevation_cluster - np.pi , elevation_cluster + np.pi, 80)  # Elevation angles
            az_grid, el_grid = np.meshgrid(azimuths, elevations)

            # Initialize the total array response
            total_response = np.zeros_like(az_grid, dtype=complex)

            # Loop over the planar array
            weight = W_mmse[:, cluster_id].reshape(Nx, Ny)
            for nx in range(Nx):
                for ny in range(Ny):
                    x = nx * dx
                    y = ny * dy
                    phase_shift = k * (x * np.sin(el_grid) * np.cos(az_grid) +
                                    y * np.sin(el_grid) * np.sin(az_grid))
                    total_response += weight[nx, ny] * np.exp(1j * phase_shift)

            # Compute the beam pattern
            beam_pattern = np.abs(total_response)**2
            beam_pattern /= np.max(beam_pattern)  # Normalize
            beam_pattern = np.abs(total_response)**2
            print(beam_pattern)
            beam_pattern = gaussian_filter(beam_pattern, sigma=20)  # Apply smoothing
            beam_pattern /= np.max(beam_pattern) 
            # Scale for visualization
            beam_pattern *= 2*1e4

            # Find main lobe index
            main_lobe_index = np.unravel_index(np.argmax(beam_pattern), beam_pattern.shape)
            main_lobe_gain = beam_pattern[main_lobe_index]

            # Convert to Cartesian coordinates
            X = beam_pattern * np.sin(el_grid) * np.cos(az_grid) + haps_x
            Y = beam_pattern * np.sin(el_grid) * np.sin(az_grid) + haps_y
            Z = beam_pattern * np.cos(el_grid) + haps_z  # For downward-facing beams

            # Main lobe facecolors: Apply cluster-specific color only to main lobe
            facecolors = np.full(beam_pattern.shape, cluster_color, dtype=object)
            facecolors[beam_pattern < 0.5 * main_lobe_gain] = 'white'  # Side lobes in default color

            # Plot the beam as a surface
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            facecolors=facecolors,
                            alpha=0.7, edgecolor='none')

        # Set labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        ax.set_title('3D Visualization of Beam Patterns with Intensity-based Lobes')
        plt.savefig("results/plot_beam/3d/beamforming_pattern_planar_3d.png")
        plt.show()

    @staticmethod
    def visualize_haps_movement_with_clusters(old_haps_position, new_haps_position, clusters_list):
        """
        Visualize the old and new positions of the HAPS with an arrow indicating movement,
        and plot the clusters on the ground.

        Parameters:
        old_haps_position (list or tuple): Old coordinates of the HAPS [x, y, z].
        new_haps_position (list or tuple): New coordinates of the HAPS [x, y, z].
        clusters_list (list): List of clusters with their ground coordinates [x, y, z].
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract old and new HAPS coordinates
        old_x, old_y, old_z = old_haps_position
        new_x, new_y, new_z = new_haps_position

        # Plot the old HAPS position
        ax.scatter(old_x, old_y, old_z, color='blue', s=100, label='Old HAPS Position', edgecolor='black')

        # Plot the new HAPS position
        ax.scatter(new_x, new_y, new_z, color='red', s=100, label='New HAPS Position', edgecolor='black')

        # Draw an arrow indicating movement
        ax.quiver(old_x, old_y, old_z, new_x - old_x, new_y - old_y, new_z - old_z,
                color='green', arrow_length_ratio=0.1, linewidth=2, label='HAPS Movement')

        # Define a list of colors for clusters
        #cluster_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']

        # Plot clusters
        for cluster_id, cluster in enumerate(clusters_list):
            cluster_x, cluster_y, cluster_z = cluster['area_coordinates']
            cluster_color = cluster_colors[cluster_id % len(cluster_colors)]  # Assign color cyclically
            ax.scatter(cluster_x, cluster_y, 0, s=80, color=cluster_color, label=f'Cluster {cluster_id+1}')  # Clusters on the ground

        # Set labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        ax.set_title('HAPS Movement and Cluster Positions')
        plt.show()
