from env.haps_parameters import PathLossParametersSBand  # Ensure this import matches your project structure
import numpy as np

class DynamicClusterInterferenceCalculator:
    def __init__(self, path_loss_calculator, transmit_power, side_lobe_gain, noise_power_density, bandwidth):
        """
        Initializes the calculator with fixed system parameters.
        
        Parameters:
        path_loss_calculator (PathLossCalculator): Instance of PathLossCalculator for path loss calculations.
        transmit_power (float): Transmit power for each cluster in Watts.
        side_lobe_gain (float): Side lobe gain as a linear ratio (not in dB).
        noise_power_density (float): Noise power density (W/Hz).
        bandwidth (float): Bandwidth in Hz.
        """
        self.path_loss_calculator = path_loss_calculator
        self.transmit_power = transmit_power
        self.side_lobe_gain = side_lobe_gain

        self.bandwidth = bandwidth
        self.clusters = []  # List to hold current clusters

    def update_clusters(self, new_clusters):
        """
        Update the clusters dynamically during the simulation.
        
        Parameters:
        new_clusters (list of dicts): List of cluster dictionaries containing 'id', 'position', and 'aod'.
        """
        self.clusters = new_clusters

    def calculate_interference(self, cluster_a, cluster_b, frequency, path_loss_parameters):
        """
        Calculate the interference power from cluster A's beam to cluster B.
        
        Parameters:
        cluster_a (dict): Dictionary containing 'id', 'position', and 'aod' for cluster A.
        cluster_b (dict): Dictionary containing 'id', 'position', and 'aod' for cluster B.
        frequency (float): Frequency in Hz.
        path_loss_parameters (PathLossParametersSBand): Parameters for path loss calculation.
        
        Returns:
        float: Interference power in Watts.
        """
        
        # Check if the clusters are adjacent based on AoD constraints
        aod_diff = abs(cluster_a['aod'] - cluster_b['aod'])
        
        
        #TODO: This threshold should be adjusted based on the beamwidth and interference constraints
        aod_threshold = 15  # degrees; adjustable based on beamwidth and interference considerations
        
        if aod_diff <= aod_threshold:
            # Calculate slant range distance
            distance = self.path_loss_calculator.slant_range(cluster_a['position'], cluster_b['position'])
            
            # Calculate path loss using the provided path loss calculator
            path_loss_dB = self.path_loss_calculator.fspl(distance)
            path_loss_linear = self.path_loss_calculator.dB_to_ratio(-path_loss_dB)
            
            # Calculate interference power at cluster B
            interference_power = self.transmit_power * self.side_lobe_gain * path_loss_linear
            return interference_power
        else:
            # No significant interference if the AoD difference is greater than the threshold
            return 0

    def calculate_total_interference(self, target_cluster_id, frequency, path_loss_parameters):
        """
        Calculate the total interference for a target cluster from all other clusters.
        
        Parameters:
        target_cluster_id (int): ID of the target cluster.
        frequency (float): Frequency in Hz.
        path_loss_parameters (PathLossParametersSBand): Parameters for path loss calculation.
        
        Returns:
        float: Total interference power in Watts for the target cluster.
        """
        total_interference = 0
        target_cluster = next(cluster for cluster in self.clusters if cluster['id'] == target_cluster_id)
        
        for cluster in self.clusters:
            if cluster['id'] != target_cluster_id:
                interference_power = self.calculate_interference(cluster, target_cluster, frequency, path_loss_parameters)
                total_interference += interference_power
        
        return total_interference
    
    
    