import numpy as np

class SmallScaleFading:
 
    def __init__(self, Nt, Nr, wavelength, K_factor=5, d_tx=0.075, d_rx=0.075):
        """
        Initializes the channel model.
        
        Parameters:
        Nt (int): Number of transmit antennas (HAPS).
        Nr (int): Number of receive antennas (Cluster).
        wavelength (float): Wavelength of the signal.
        K_factor (float): Rician factor, representing the LOS/NLOS ratio.
        d_tx (float): Spacing between transmit antennas (default half-wavelength).
        d_rx (float): Spacing between receive antennas (default half-wavelength).
        """
        self.Nt = Nt
        self.Nr = Nr
        self.wavelength = wavelength
        self.K_factor = K_factor
        self.d_tx = d_tx
        self.d_rx = d_rx

    def calculate_angles(self, tx_pos, rx_pos):
        """
        Calculate azimuth and elevation AoD and AoA based on transmitter and receiver positions.
        
        Parameters:
        tx_pos (tuple): Coordinates of the transmitter (x, y, z).
        rx_pos (tuple): Coordinates of the receiver (x, y, z).
        
        Returns:
        (tuple): Azimuth and elevation angles for AoD and AoA.
        """
        # Calculate horizontal distance and height difference
        d_horizontal = np.sqrt((rx_pos[0] - tx_pos[0]) ** 2 + (rx_pos[1] - tx_pos[1]) ** 2)
        height_diff = tx_pos[2] - rx_pos[2]
        #print(tx_pos, rx_pos)
        # Calculate Azimuth AoD and AoA
        azimuth_aod = np.arctan2(rx_pos[1] - tx_pos[1], rx_pos[0] - tx_pos[0])
        azimuth_aoa = azimuth_aod  # same direction

        # Calculate Elevation AoD and AoA
        elevation_aod =   np.arctan2(d_horizontal, height_diff)
        #print("elevation_aod", np.degrees(elevation_aod))
        elevation_aoa = elevation_aod  # same direction

        return azimuth_aod, elevation_aod, azimuth_aoa, elevation_aoa

    def array_response(self, N, d, wavelength, azimuth, elevation):
        """
        Calculate the array response vector for a linear array.
        
        Parameters:
        N (int): Number of antennas in the array.
        d (float): Spacing between antennas.
        wavelength (float): Wavelength of the signal.
        azimuth (float): Azimuth angle (in radians).
        elevation (float): Elevation angle (in radians).
        
        Returns:
        np.ndarray: Array response vector.
        """
        k = 2 * np.pi / wavelength
        response = np.exp(1j * k * d * np.arange(N) * np.sin(elevation) * np.cos(azimuth))
        return response
    def array_response_grid(self, N, d, wavelength, azimuth, elevation):
        """
        Calculate the array response vector for a linear array.
        
        Parameters:
        N (int): Number of antennas in the array.
        d (float): Spacing between antennas.
        wavelength (float): Wavelength of the signal.
        azimuth (float): Azimuth angle (in radians).
        elevation (float): Elevation angle (in radians).
        
        Returns:
        np.ndarray: Array response vector.
        """
        k = 2 * np.pi / wavelength
        d_x = d  
        d_y = d
        M = 4
        N = 4
        response = np.zeros((M, N), dtype=complex)
        for m in range(M):
            for n in range(N):
                phase_shift = k * (m * d_x * np.sin(elevation) * np.cos(azimuth) +
                               n * d_y * np.sin(elevation) * np.sin(azimuth))
                response[m, n] = np.exp(1j * phase_shift)
        #response = np.exp(1j * k * d * np.arange(N) * np.sin(elevation) * np.cos(azimuth))
        #print(response.flatten())
        return response.flatten()
    def calculate_H_LOS(self, tx_pos, rx_pos):
        """
        Calculate the LOS component of the channel matrix.
        
        Parameters:
        tx_pos (tuple): Coordinates of the transmitter (x, y, z).
        rx_pos (tuple): Coordinates of the receiver (x, y, z).
        
        Returns:
        np.ndarray: LOS channel matrix.
        """
        azimuth_aod, elevation_aod, azimuth_aoa, elevation_aoa = self.calculate_angles(tx_pos, rx_pos)
        a_tx = self.array_response_grid(self.Nt, self.d_tx, self.wavelength, azimuth_aod, elevation_aod)
        a_rx = self.array_response(self.Nr, self.d_rx, self.wavelength, azimuth_aoa, elevation_aoa)
        distance_3d = np.sqrt((rx_pos[0] - tx_pos[0])**2 + (rx_pos[1] - tx_pos[1])**2 + (rx_pos[2] - tx_pos[2])**2)
        # Outer product to form the LOS channel matrix
        
        H_LOS = np.outer(a_rx, a_tx.conj()) * np.exp( - 1j * 2 * np.pi * distance_3d / self.wavelength)
        
        return H_LOS

    def calculate_H_NLOS(self):
        """
        Calculate the NLOS component of the channel matrix.
        
        Returns:
        np.ndarray: NLOS channel matrix with Rayleigh fading.
        """
        H_NLOS = (np.random.normal(0, 1, (self.Nr, self.Nt)) + 
                  1j * np.random.normal(0, 1, (self.Nr, self.Nt))) / np.sqrt(2)
        return H_NLOS

    def calculate_total_H(self, tx_pos, rx_pos, LOS_condition): 
        """
        Calculate the total channel matrix combining LOS and NLOS components.
        
        Parameters:
        tx_pos (tuple): Coordinates of the transmitter (x, y, z).
        rx_pos (tuple): Coordinates of the receiver (x, y, z).
        
        Returns:
        np.ndarray: Total channel matrix.
        """
        H_NLOS = self.calculate_H_NLOS()
        #print("HNLOS", H_NLOS, H_NLOS.shape)
        if LOS_condition:  
            H_LOS = self.calculate_H_LOS(tx_pos, rx_pos)
            #print("HLOS", H_LOS, H_LOS.shape)
            # Combine LOS and NLOS with the Rician K factor
            H = np.sqrt(self.K_factor / (self.K_factor + 1)) * H_LOS + np.sqrt(1 / (self.K_factor + 1)) * H_NLOS
            #print("H", H, H.shape)
            return H.reshape(-1)
        else : 
            return H_NLOS.reshape(-1)
    
