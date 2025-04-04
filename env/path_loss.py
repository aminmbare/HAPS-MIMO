import numpy as np 
from scipy.stats import norm
import math 
import random
from typing import List
from env.haps_parameters import PathLossParametersSBand


class PathLossCalculator:
    def __init__(self, fc: float):
        """
        Initializes the path loss calculator with the carrier frequency.
        :param fc: Carrier frequency in GHz.
        """
        self.fc = fc
  
      
    def slant_range(self, area_position: List, Haps_position : List) -> float:
        """
        Calculate the slant range distance (d) between the satellite/HAPS and ground terminal.
        :param elevation_angle_deg: Elevation angle in degrees.
        :return: Slant range distance in km.
        """
        
        delta_x = np.abs(area_position[0] - Haps_position[0])
        delta_y = np.abs(area_position[1] - Haps_position[1])
        delta_z = np.abs(area_position[2] - Haps_position[2])
        d = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        return d

    def fspl(self, d: float) -> float:
        """
        Calculate the Free Space Path Loss (FSPL).
        :param d: Slant range distance in m.
        :return: FSPL in dB.
        """
        return 32.45 + 20 * math.log10(self.fc) + 20 * math.log10(d)

    def generate_shadow_fading(self, los: bool, parameters: PathLossParametersSBand) -> float:
        """
        Generate a random shadow fading value based on normal distribution.
        :param los: True if LOS, False if NLOS.
        :param parameters: Path loss parameters containing standard deviations for SF.
        :return: Shadow fading in dB.
        """
        sigma_sf = parameters.los_sf if los else parameters.nlos_sf
        return random.gauss(0, sigma_sf)  # Mean 0, standard deviation σ_SF

    def generate_clutter_loss(self,los: bool, parameters: PathLossParametersSBand) -> float:
        """
        Generate a random clutter loss value based on normal distribution.
        :param parameters: Path loss parameters containing standard deviation for CL.
        :return: Clutter loss in dB.
        """
        
        return  0 if los  else  parameters.nlos_cl   

    def calculate_path_loss(self,area_position, haps_position, elevation_angle_deg: float, los: bool, parameters: PathLossParametersSBand, indoor_user_percentage : float , traditional_building_percentage : float) -> float:
        """
        Calculate the basic path loss (PL_b).
        :param elevation_angle_deg: Elevation angle in degrees.
        :param los: True if LOS, False if NLOS.
        :param parameters: Path loss parameters containing standard deviations for SF and CL.
        :return: Path loss in dB.
        """
        # Calculate slant range distance
        d = self.slant_range(area_position, haps_position)

        # Calculate FSPL
        fspl_value = self.fspl(d)

        # Generate shadow fading and clutter loss based on distributions
        shadow_fading = self.generate_shadow_fading(los, parameters)
        clutter_loss = self.generate_clutter_loss(0,parameters)
        if los:
            building_penetration_loss = 0
        else : 
            t_building = 0 if random.random() < traditional_building_percentage else 1
            building_penetration_loss = self.calculate_building_penetration_loss(elevation_angle_deg, t_building, 0.5) 
        # Total path loss
        path_loss = fspl_value + shadow_fading + clutter_loss + building_penetration_loss
        return path_loss

    
    def calculate_building_penetration_loss(self,
        elevation, 
        t_building, # the type of building, 0 for traditional, 1 for thermally-efficient
        p_not_exceed 
        
    ):
        # parameters for buildings
        # (traditional, thermally-efficient)
        r = (12.64, 28.19)
        s = (3.72, -3.0)
        t = (0.96, 8.48)
        u = (9.6, 13.5)
        v = (2.0, 3.8)
        w = (9.1, 27.8)
        x = (-3.0, -2.9)
        y = (4.5, 9.4)
        z = (-2.0, -2.1)
        f = self.fc
        theta = elevation - 90 # elevation angle of the path at the building façade (degrees)
        P = p_not_exceed # probability that loss is not exceeded (0.0 < P < 1.0) !!!
        F_minus_1_P = norm.ppf(P) # inverse cumulative normal distribution as a function of probability

        L_e = 0.212 * abs(theta)
        L_h = r[t_building] + s[t_building] * np.log10(f) + t[t_building] * (np.log10(f) ** 2)

        miu_1 = L_h + L_e
        miu_2 = w[t_building] + x[t_building] * np.log10(f)
        sigma_1 = u[t_building] + v[t_building] * np.log10(f)
        sigma_2 = y[t_building] + z[t_building] * np.log10(f)

        A_P = F_minus_1_P * sigma_1 + miu_1
        B_P = F_minus_1_P * sigma_2 + miu_2
        C = -3.0

        L_BEL_P = 10 * np.log10(
            (10 ** (0.1 * A_P)) + 
            (10 ** (0.1 * B_P)) +
            (10 ** (0.1 * C))
        ) # building entry loss not exceeded for the probability P, dB

        return L_BEL_P            
@staticmethod  
def dB_to_ratio(value_dB):
    value_ratio = 10 ** (value_dB / 10)
    return value_ratio
@staticmethod
def ratio_to_dB(value_ratio):
    value_dB = 10 * np.log10(value_ratio)
    return value_dB