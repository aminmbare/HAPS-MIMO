import pandas as pd
from datetime import timedelta
import numpy as np
import tqdm
from typing import Optional, List
import logging
from env.geo_utils import *
from env.plotting_utils  import *
import os
from sklearn.cluster import KMeans
from itertools import chain
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from env.Map import Map
import datetime
from env.ChannelModel import MU_MIMO_CapacityCalculator
from env.small_scale_fading import SmallScaleFading
from env.path_loss import PathLossCalculator
from scipy.optimize import minimize
from itertools import combinations
from copy import deepcopy
from pyswarm import pso
from shapely import wkt
from sklearn.cluster import DBSCAN

class HapsController(object):
    def __init__(self,haps_sband_parameters,AreasList : list, n_areas : int, beta: float, gamma : float, alpha : float = 0.92, sigma : float = 0.4, algo : str = "rl" )-> None:
        self.HAPS_Channel_model = MU_MIMO_CapacityCalculator(SmallScaleFading(16,1,0.15), PathLossCalculator(2),-100, 20)
        self.haps_sband_parameters = haps_sband_parameters  
        self.areas = AreasList
        areas_name = [area.area_name for area in self.areas]
        self.area_type_array = np.array([area.area_type for area in self.areas])
        self.indoor_users_array = np.array([area.indoor_users for area in self.areas])
        self.traditional_building_percentage_array = np.array([area.traditional_building_percentage for area in self.areas])
        self.Values_df = pd.DataFrame(columns = areas_name)
        self.n_areas = n_areas
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.area_names = areas_name
        self.areas_chosen_list = [] 
        self.N = pd.DataFrame(columns = areas_name)
        self.means = pd.DataFrame(columns = areas_name)
        self.variances = pd.DataFrame(columns = areas_name)
        self.milan_map = Map()
        self.algo = algo
        self.min_capacity = 50
        

    def create_Values_dataframe(self) -> None:
        
        for area in tqdm.tqdm(self.areas, desc = "Creating Values dataframe", total = len(self.areas)):
            area_stats = area.get_area_stats()
            #print(area_stats)
            self.Values_df[area.area_name] = area_stats["integral"]
            self.N[area.area_name] = area_stats["integral"]
            self.means[area.area_name]= area_stats["integral"]
            self.variances[area.area_name] = area_stats["integral"]
            
            
        self.Values_df.fillna(0, inplace = True)
        self.N.fillna(1, inplace = True)
        self.means.fillna(0, inplace = True)
        self.variances.fillna(1, inplace = True)
        self.N.loc[:,:] = 1
        self.means.loc[:,:] = 10
        self.variances.loc[:,:] = 1
        if self.algo == "epsilon-greedy": 
            self.Values_df.loc[:,:] = 0
        if self.algo == "exp":
            self.Values_df.loc[:,:] = 1
    def create_energy_Values_dataframe(self) -> None:
        for area in tqdm.tqdm(self.areas, desc = "Creating Energy Values dataframe", total = len(self.areas)):
            area_stats = area.get_area_energy_stats()
            self.Energy_Values_df[area.area_name] = area_stats["integral"]
        self.Energy_Values_df.fillna(0, inplace = True)


    def getSimulationTime(self)-> int:
        return len(self.areas[0][0])


    def getTrainTime(self) -> int : 
        return self.areas[0][0]._train_len


    def SetSimulationTimer(self)-> None:
        Simulation_time = len(self.areas[0][0])
        print(Simulation_time)
        self.time = self.areas[0][0].time[-Simulation_time:]
        self.test_len = self.areas[0][0]._test_len
        self.test_time = self.time[-self.test_len:]
    def __len__(self)-> int:
        return len(self.areas)
    
    
    def grid_definition(self, grid_size : int)-> None: 
        x_min = max(self.haps_position[0] - 10_000, self.haps_boundaries[0])
        x_max = min(self.haps_position[0] + 10_000, self.haps_boundaries[2])
        y_min = max(self.haps_position[1] - 10_000, self.haps_boundaries[1])
        y_max = min(self.haps_position[1] + 10_000, self.haps_boundaries[3])
        x_range = np.linspace(x_min, x_max, grid_size)
        y_range = np.linspace(y_min, y_max, grid_size)
        self.grid_size = grid_size 
        self.X, self.Y = np.meshgrid(x_range, y_range)
        print(self.X.shape, self.Y.shape)
    def init_haps_position(self,x_coor, y_coor): 
        x_position = self.X[x_coor, y_coor]
        y_position = self.Y[x_coor, y_coor]
        self.haps_position = [x_position, y_position]
        self.haps_position_init = deepcopy(self.haps_position)
           
    def init_haps_position_altenate(self, morning_coor, evening_coor):
        x_position = self.X[morning_coor[0], morning_coor[1]]
        y_position = self.Y[morning_coor[0], morning_coor[1]]
        haps_position = [x_position, y_position]
        self.haps_position_morning = deepcopy(haps_position)
        x_position = self.X[evening_coor[0], evening_coor[1]]
        y_position = self.Y[evening_coor[0], evening_coor[1]]
        self.haps_position_night = [x_position, y_position]
        self.haps_position_alter = [self.haps_position_morning, self.haps_position_night]
        self.haps_position = self.haps_position_alter[0]
    def reset(self, cs_number : int, dist : int)-> None:
        self.Dmax = dist
        self.inversion_list = []
        self.off_load_history = []
        self.create_Values_dataframe()
        self.hourly_ps_buffer = np.zeros(len(self.areas))
        #self.create_energy_Values_dataframe()
        if self.algo == "exp": 
            self.probs_exp3 = np.zeros(len(self.areas))
        self.count_offload = 0
        self.tot_count = 0
        self.capacity_utilization = np.zeros(self.getSimulationTime())
        self.area_names = [area.area_name for area in self.areas]
        self.SetSimulationTimer()
        for area in self.areas:
            area.reset()
            area.create_beam_data_frame(self.test_time)
        self.milan_map.create_milan_coordinates()
        _,self.haps_boundaries = self.milan_map.Areas_Positions()
        print("Bounds",self.haps_boundaries)
        self.area_centorid = pd.read_csv(f"results/cs/clustered_areas_r_{cs_number}.csv")
        self.area_centorid["geometry"] = self.area_centorid['geometry'].apply(wkt.loads)
        self.area_centorid["centroid"] = self.area_centorid["centroid"].apply(wkt.loads)
        self.HAPS_Channel_model.set_SINR_map(self.haps_boundaries)
        print(self.area_centorid)
        #haps_position_list = [ [1023013.1629615711, 5692317.35968741],
        #                [1018462.7214474182, 5696839.549812133],
        #                [1018430.2574162285, 5698423.859574158], 
        #                [1019655.9246205585, 5697160.293967155],
        #                [1015886.2123675257, 5698318.799039762]]
        ##[1.01773703 5.70264913]
        #[1.01590405 5.7016425 ]
        ## ld 2
        #[1.01161469 5.70062607]
        #[1.02083763 5.69791268]
        #[1.01995776 5.69726707
        #[1.02083763 5.69791268]
        #  [1024688.4697913013, 5692291.653335674] fffff r = 0 night
        haps_position_list = [ [1025115.8519903885, 5698649.074913133]
                               , [1020850.8115355365, 5698037.737636596], 
                               [1025464.0999923623, 5704047.716536544], 
                               [1016295.3471440761, 5705563.011859086], 
                               [1016673.2340624549, 5692845.10228333], 
                              [1025115.8519903885, 5698649.074913133], 
                                [1020850.8115355365, 5698037.737636596], 
                                [1025464.0999923623, 5704047.716536544], 
                                [1016295.3471440761, 5705563.011859086], 
                                [1016673.2340624549, 5692845.10228333] 
                                ]
        ###self.haps_position = [(self.haps_boundaries[0]+self.haps_boundaries[2])/2, (self.haps_boundaries[1]+self.haps_boundaries[3])/2]
        self.haps_position_alter_ld_2 = [[1017741.2605456186, 5698682.914792892], [1022061.0223174309, 5690400.088365226]]
        self.haps_position_alter_r_0 = [[1023867.4046088054, 5698288.981563782],[1022042.7036355805, 5696832.699095069] ]
        self.haps_position_alter = [ [1016402.6469052962, 5697453.970108161], [1020057.3386730158, 5697889.461336695] ]
        #self.haps_position = haps_position_list[cs_number]
        #self.haps_position_alter = [[1020850.8115355365, 5698037.737636596] , [1020850.8115355365, 5698037.737636596]]
        self.haps_position = haps_position_list[cs_number]
        #self.haps_position = 
        self.haps_position_init = deepcopy(self.haps_position)
        print("boundaries", self.haps_boundaries)
        #print(self.haps_position)
        #self.haps_position = [1022074.161790212, 5693875.259116972]
        print(self.area_centorid)
        #if self.algo == "epsilon-greedy": 
        #    self.N = np.ones(len(self.areas))
        self.los_df = pd.DataFrame(columns = self.test_time , index = self.area_names)
        self.los_df.fillna(0.0, inplace = True)
        self.load_history = pd.DataFrame(columns = self.test_time , index = ["load"])
        self.load_history.fillna(0.0, inplace = True)
        self.haps_positions_df = pd.DataFrame(columns = ["x", "y"], index = self.test_time)
        self.haps_positions_df.fillna(0.0, inplace = True)
        self.haps_beams_load = pd.DataFrame(columns = self.test_time, index = list(range(self.n_areas)))
        self.haps_beams_load.fillna(0.0, inplace = True)
        self.haps_capacity_per_beam = pd.DataFrame(columns = self.test_time, index = list(range(self.n_areas)))
        self.haps_capacity_per_beam.fillna(0.0, inplace = True)
        self.Area_selection_df = pd.DataFrame(columns = self.test_time, index = list(range(len(self.areas))))
        self.Area_selection_df.fillna(0.0, inplace = True)
        self.n_areas_history = []
        self.area_selection_array = np.zeros(len(self.areas))
        self.reward_error_list = list()

    def get_values_per_time(self, time_int: int) -> np.ndarray:
        def get_part_of_day(time):
            hour = time.hour
            if 8 <= hour < 16:
                return 'Day'
            elif 16 <= hour <= 23:
                return 'Night'
            
        time = self.time[time_int]
        day = time.day_name()
        part_of_day = get_part_of_day(time)
        day_part_of_day = day + "-" + part_of_day
        # Get the previous day's name
        previous_day = (time - timedelta(days=1)).day_name()
        previous_day_part_of_day = previous_day + "-" + part_of_day

        logging.info(f" >>>>>>  part of day: {day_part_of_day}, time : {time}")
        if self.algo == "exp3" : 
            values = self.Values_df.loc[day_part_of_day].values
            previous_day_valyes = self.Values_df.loc[previous_day_part_of_day].values
            for i , area in enumerate(self.areas):
                x = 0
                
                for j,ps in enumerate(area.hps_buffer) : 
                    if j == 0:
                        
                        x += ps
                    else : 
                        x += pow(self.beta, j)*ps
                #print(values[i], x)
                values[i] = values[i] +  np.sqrt( 0.8*previous_day_valyes[i]+ x) 
            #values =  0.3*self.Values_df.loc[day_part_of_day].values + self.hourly_ps_buffer
            #if part_of_day == "Night":
            #    theta = time.hour - 15
            #    previous_influence = self.Values_df.loc[day + "-Day"].values
            #    values = values + pow(self.beta,theta)*previous_influence
        elif self.algo == "greedy"   or self.algo == "epsilon-greedy": 
            values = self.Values_df.loc[day_part_of_day].values
        if self.algo == "ucb":
            values = np.zeros(len(self.areas))
            for i, area in enumerate(self.areas):
                N = self.N.loc[day_part_of_day, area.area_name]
                if N < 1:
                    values[i] = 1e400
                    continue 
                
                values[i] =  np.mean(area.hps_window[day_part_of_day]) + np.sqrt(2*np.log(time_int)/N)
                if len(area.hps_window[day_part_of_day]) == 0:
                    values[i] = 1e400
        elif self.algo == "exp": 
            #values = self.Values_df.loc[day_part_of_day].values
            #values = (1-self.alpha)* values / np.sum(values) + self.alpha/len(self.areas)
            means = self.means.loc[day_part_of_day].values
            print(means)
            variances = self.variances.loc[day_part_of_day].values
            print(variances)
            values = np.random.normal(means, np.sqrt(variances))
            
        return values

    def update_areas_ps_buffer(self, time_int : int)-> None:
        for area in self.areas:
            area.update_ps_buffer_area(time_int)


    def select_values_adaptive(self,values, z_threshold=0):
    # Convert values to a numpy array
        values = np.array(values)
        
        # Get the sorted indexes in descending order
        sorted_indexes = np.argsort(values)[::-1]
        
        # Sort values in descending order using the sorted indexes
        sorted_values = values[sorted_indexes]
        #print("sorted values", sorted_values)
        # Calculate mean and standard deviation of the sorted values
        mean = np.mean(sorted_values)
        std = np.std(sorted_values)
        #print("sorted indexes", sorted_indexes)
        # Calculate z-scores
        z_scores = (sorted_values - mean) / std
        print(z_scores)
        #logging.info(f"Z-scores: {z_scores}")
        # Select indexes with z-scores above the threshold
        selected_indexes = sorted_indexes[z_scores > z_threshold]
        #if len(selected_indexes) > 6 : 
        #    selected_indexes = selected_indexes[:6]
        
        #self.areas_chosen_list.append(len(selected_indexes))
        #Choosen_values = values[selected_indexes]
        return selected_indexes 

    
    def get_policy(self, time_int : int, method : bool = True) -> np.ndarray:
        #print(f"time int: {self.time[time_int]}")
        values = self.get_values_per_time(time_int)
        if method : 
        
            logging.info(values)
              
            values_sort_index = np.argsort(values)[::-1]
            ChosenAreas = values_sort_index[:self.n_areas]
            ChosenAreas_values = values[ChosenAreas]

        else : 
            
            ChosenAreas, ChosenAreas_values = self.select_values_adaptive(values)
        if self.algo == "epsilon-greedy": 
            #if time_int  < self.getTrainTime() : 
            print("exploration")
            ChosenAreas = random.choices(np.arange(16), k = self.n_areas)
            self.ChosenAreas = ChosenAreas
            #else : 
            #    print("exploitation")
            #    values_sort_index = np.argsort(values)[::-1]
            #    ChosenAreas = values_sort_index[:self.n_areas]
            #    self.ChosenAreas = ChosenAreas
            #    ChosenAreas_values = values[ChosenAreas]
        logging.info(f"Areas chosen: {ChosenAreas}")
        logging.info(f" Q(s,a) : {values}")
        
        if self.algo == "exp": 
            
            self.probs_exp3= (1 - self.gamma) *  values / np.sum(values) + self.gamma / len(self.areas)
            print(values)
            print(self.probs_exp3)
            ChosenAreas =  np.random.choice(np.arange(len(self.areas)), size=self.n_areas, replace=False, p=self.probs_exp3) 
            ChosenAreas_values = values[ChosenAreas]
            self.ChosenAreas = ChosenAreas
            
        #if self.algo =="ucb": 
        #    time = self.time[time_int]
        #    def get_part_of_day(time):
        #        hour = time.hour
        #        if 8 <= hour < 16:
        #            return 'Day'
        #        elif 16 <= hour <= 23:
        #            return 'Night'
        #
        #    part_of_day = get_part_of_day(time)
        #    day = time.day_name()
        #    state = day + "-" + part_of_day
        #    values = values + np.sqrt(2*np.log(time_int)/self.N.loc[state].values)
        #    values_sort_index = np.argsort(values)[::-1]
        #    ChosenAreas = values_sort_index[:self.n_areas]
        #    ChosenAreas_values = values[ChosenAreas]
        #    self.ChosenAreas = ChosenAreas
        
        return  ChosenAreas, ChosenAreas_values, values


        logging.debug(f"Values: {values}, shape: {values.shape}")
        logging.debug(f"Predictions: {predictions}, shape: {predictions.shape}")
        logging.debug(f"Old OPS: {old_ops}, shape: {old_ops.shape}")
        logging.info(values)
        logging.info(predictions)
        policy = np.exp(values+1) + self.alpha * predictions + self.beta * old_ops
        logging.info(f"Policy: {policy/policy.sum()}")
        logging.info(f"Policy Entropy: {entropy(policy,base = 2)}")
        policy_sort_index = np.argsort(policy)[::-1]
        priority_indexes = policy_sort_index[:5]
        low_priority_area = np.random.choice(policy_sort_index[5:], 1 ,replace = False)
        ChosenAreas = np.concatenate((priority_indexes,low_priority_area))
        
        logging.info(f"Areas chosen in the loterry: {ChosenAreas}")
        policy = policy[ChosenAreas]
        policy = policy / policy.sum()
        logging.info(f"Policy: {policy}")
        return policy, ChosenAreas
    def create_SINR_map(self, path)-> None:
        cluster_list = []
        for idx , row in self.area_centorid.iterrows():
            cluster_data = dict()
            cluster_data["boundaries"] = row.geometry
            cluster_data["centroid"] = row.centroid
            cluster_data["area_name"] = row.area_name
            cluster_list.append(cluster_data)
        self.HAPS_Channel_model.compute_SINR_map(cluster_list, self.area_selection_array, path)
        
    def choose_action(self, time_int : int) -> Optional[List]:
        time = self.time[time_int]
        logging.info(f"Time : {time}")
        
        print(f"Time : {time}")
        
        #self.find_haps_positio_static(time_int, True)
        #self.find_haps_positio_static(time_int, False)
        
        if time.hour >= 0 and time.hour < 8:
            return None
        ChosenAreas, ChosenAreas_values, values = self.get_policy(time_int)
        #self.ChosenAreas = ChosenAreas
        print(f"Chosen Areas: {ChosenAreas}")
        #self.Area_selection_df.iloc[ChosenAreas, time_int] = 1 
        logging.info(f"Chosen Areas Values: {ChosenAreas_values}")
        logging.info(f"Chosen Areas: {ChosenAreas}")
        #ChosenAreas = [0,1,2,3,6,11,14,15]
        #ChosenAreas = [0,3,6,8,9,11,14,15]

        choosen_areas_centroid = self.area_centorid.iloc[ChosenAreas]
        ##ChosenAreas = np.array(ChosenAreas)[AreasIndex]
        ##print(choosen_areas_centroid.loc[0].centroid.x). 
        if time_int > self.getTrainTime():
            time_temp = time_int - self.getTrainTime() 
            
            if not hasattr(self, 'ChosenAreas'):
                    #print("Chosen Areas", self.ChosenAreas)
                    #self.find_haps_position(choosen_areas_centroid, time_int, True)
                    ChosenAreas = self.position_controller(choosen_areas_centroid,np.array(ChosenAreas_values), np.array(ChosenAreas), True)
                    
            else : 
                sorted_previous_choosen_areas = sorted(self.ChosenAreas)
                sorted_current_choosen_areas = sorted(ChosenAreas)
                if sorted_previous_choosen_areas == sorted_current_choosen_areas:
                    if min(self.final_capacities) < 5 : 
                       ChosenAreas = self.position_controller(choosen_areas_centroid,np.array(ChosenAreas_values), np.array(ChosenAreas), True)
                    else :
                       #ChosenAreas = self.position_controller(choosen_areas_centroid,np.array(ChosenAreas_values), np.array(ChosenAreas), True)
                        self.find_haps_position(choosen_areas_centroid,ChosenAreas, time_temp , False)
                    #ChosenAreas = self.position_controller(choosen_areas_centroid,np.array(ChosenAreas_values), np.array(ChosenAreas), True)
                else :  
                    #ChosenAreas = self.position_controller(choosen_areas_centroid,values, ChosenAreas, False)
                    ChosenAreas = self.position_controller(choosen_areas_centroid,np.array(ChosenAreas_values), np.array(ChosenAreas), True)
        ####
            #self.ChosenAreas = deepcopy(ChosenAreas)
            #self.find_haps_position(choosen_areas_centroid, time_int, True)
            
            #self.find_haps_position(choosen_areas_centroid, ChosenAreas,time_temp, False)
            self.traffic_report(time_int, ChosenAreas) 
            self.position_report(time_temp)    
            #self.ChosenAreas = deepcopy(ChosenAreas)     
            self.area_selection_array[ChosenAreas] += 1 
            for i in self.ChosenAreas: 
                #area_name = self.area_centorid.loc[i].area_name
                self.Area_selection_df.iloc[i, time_temp] = 1
        self.ChosenAreas = ChosenAreas
        #self.find_haps_position(choosen_areas_centroid, time_int, False)     
        if time.hour == 23 :
            self.haps_position = self.haps_position_init
        #if time.hour == 15 : 
        #    self.haps_position = self.haps_position_alter[1]
        
        #self.n_areas = 6 
        #print("time int", time_int)
        #self.find_haps_positio_static(time_int)
        ###self.find_haps_position(ChosenAreas_values, choosen_areas_centroid, time)
        #self.find_haps_position(choosen_areas_centroid, time_int)
        #self.find_haps_positio_static(time_int)
        return ChosenAreas
        

        #policy,ChosenAreas = self.get_policy(time_int)
        self.ChosenAreas = list(range(len(self.areas)))
        
        #actions = np.random.choice(ChosenAreas,self.n_areas,replace = False , p = policy)
        #actions = self.ChosenAreas[np.argsort(policy)[::-1]][:self.n_areas]
        actions  = np.random.choice(self.ChosenAreas,self.n_areas,replace = False)
        return actions
    
    
    def traffic_report(self,time_int : int, chosen_areas : Optional[List])-> None: 
        if chosen_areas is None:
             logging.info("HAPS is not active at this time\n")
    
        else :
            log_message = ", ".join([str(i) for i in chosen_areas])
            logging.info(f"Chosen Areas: {log_message}\n")

        print(f"Chosen Areas: {chosen_areas}")
        off_load = np.zeros(len(self.areas))
        peak = np.zeros(len(self.areas))
        load_redcution_ran = np.zeros(len(self.areas))
        ideal_peak_hourly = list()
        tot_off_load = 0
        #print("chosen areas", chosen_areas)
        self.haps_traffic = np.zeros(self.n_areas)
        choosen_areas_array = np.array(chosen_areas)
        for i, area in enumerate(self.areas):
           bs_with_peaks,off_bound_traffic, load_redcution_area  = area.post_processing(time_int,
                                         i in chosen_areas if chosen_areas is not None else False)
           
           if chosen_areas is not None:
            ideal_peak_hourly.append(bs_with_peaks)
            logging.info(f"Area {i} : {area.area_name}")
            logging.info(f"Base Stations with peaks: {bs_with_peaks}")
            logging.info(f"Off bound Load: {off_bound_traffic:e}")
            off_load[i] = off_bound_traffic
            peak[i] = bs_with_peaks
            load_redcution_ran[i] = load_redcution_area
            choosen_area_index = np.where(choosen_areas_array == i)[0]
            #print("haps traffic", self.haps_traffic)
            #print("choosen area index", choosen_area_index, "choosen areas", choosen_areas_array)
            
            self.haps_traffic[choosen_area_index] = off_bound_traffic
            #self.Area_selection_df.loc[area.area_name, time_int] = 1
            logging.info("\n")
        
            
            tot_off_load += off_bound_traffic
            area.update_area_traffic(time_int)
        
        ideal_areas = np.argsort(load_redcution_ran)[::-1][:self.n_areas]
        #logging.info(f"Ideal Area: {ideal_areas}")
        for i, area in enumerate(self.areas):
            if i in  ideal_areas:
               
               area.area_ideal_decision(time_int,True)
            else:
               area.area_ideal_decision(time_int,False)

        #if chosen_areas is not None:
        #    self.off_load_history.append(tot_off_load) 
        #    self.capacity_utilization[time_int] = tot_off_load/self.haps_capacity
    def save_cap_stat(self, path : str)-> None:
        for area in self.areas:
            #save excess bs  lisr
            np.save(os.path.join(path, f"{area.area_name}_excess_bs.npy"), area.excess_bs_list)
            #save bs traffic
            np.save(os.path.join(path, f"{area.area_name}_bs_traffic.npy"), area.excess_traffic_list)
            ## after off loading (ol)
            np.save(os.path.join(path, f"{area.area_name}_excess_bs_ol.npy"), area.excess_bs_list_ol)
            ## save area_tot_traffic
            np.save(os.path.join(path, f"{area.area_name}_area_tot_traffic.npy"), area.area_tot_traffic)
            for i,bs in enumerate(area): 
                np.save(os.path.join(path, f"{area.area_name}_BS_{i}_ol.npy"), bs.old_load) 
                np.save(os.path.join(path, f"{area.area_name}_BS_{i}_nl.npy"), bs.new_load) 
            
            
    def bs_load_data(self, path : str)-> None:
        for  area in self.areas:
            area.save_bs_load(path)

    def peak_occurence_csv_save(self, path)-> None:
            self.capacity_utilization[time_int] = tot_off_load/self.haps_capacity
        
    def bs_load_data(self, path : str)-> None:
        for  area in self.areas:
            area.save_bs_load(path)
    
    def peak_occurence_csv_save(self, path)-> None:
        peak_occurence_df = pd.DataFrame(columns = ["time"]+ self.area_names)
        peak_occurence_ol_df = pd.DataFrame(columns = ["time"]+ self.area_names)
        peak_occurence_ideal_df = pd.DataFrame(columns = ["time"]+ self.area_names)
        simulation_time = self.getSimulationTime()
        for area in self.areas: 
            peak_occurence , peak_occurence_ol, peak_occurence_ideal = area.get_area_peak_occurence_report(simulation_time)
            peak_occurence_df[area.area_name] = peak_occurence
            peak_occurence_ol_df[area.area_name] = peak_occurence_ol
            peak_occurence_ideal_df[area.area_name] = peak_occurence_ideal

    
        peak_occurence_ol_df["time"] = pd.to_datetime(self.time)
        peak_occurence_df["time"] = pd.to_datetime(self.time)
        peak_occurence_ideal_df["time"] = pd.to_datetime(self.time)
        
        peak_occurence_ol_df.to_csv(os.path.join(path,"peak_occ_ol.csv"))
        peak_occurence_df.to_csv(os.path.join(path,"peak_occ.csv"))
        peak_occurence_ideal_df.to_csv(os.path.join(path,"peak_occ_ideal.csv"))
    
    
    def peaks_csv_save(self, path)-> None:
        peak_occurence_df = pd.DataFrame(columns = ["time"]+ self.area_names)
        peak_occurence_ol_df = pd.DataFrame(columns = ["time"]+ self.area_names)
        
        simulation_time = self.getSimulationTime()
        for area in self.areas: 
            peak_occurence , peak_occurence_ol = area.get_area_peak_report(simulation_time)
            peak_occurence_df[area.area_name] = peak_occurence
            peak_occurence_ol_df[area.area_name] = peak_occurence_ol
            
        peak_occurence_ol_df["time"] = pd.to_datetime(self.time)
        peak_occurence_df["time"] = pd.to_datetime(self.time)
        
        peak_occurence_ol_df.to_csv(os.path.join(path,"peak_ol.csv"))
        peak_occurence_df.to_csv(os.path.join(path,"peak.csv"))
        
    
    
    def save_haps_utilization(self,path : str)-> None:
        haps_utilization_df = pd.DataFrame(columns = ["time","load"])
        haps_utilization_df["time"] = pd.to_datetime(self.time)
        haps_utilization_df["load"] = self.capacity_utilization
        haps_utilization_df.to_csv(os.path.join(path,"haps_utilization.csv"))


    def final_report(self,plot_bs: bool, stat_plot : bool, path : str) -> np.ndarray: 
            
        old_load_dist = np.zeros(len(self.areas))
        new_load_dist = np.zeros(len(self.areas))
        ideal_load_dist = np.zeros(len(self.areas))
        old_traffic_load_per_area = list()
        new_traffic_load_per_area = list()
        ideal_traffic_load_per_area = list()
        
        traffic_old_list = list()
        traffic_new_list = list()
        new_traffics_sum = list()
        traffic_sum = list()
        total_new_traffic = 0
        total_old_traffic = 0
        total_ideal_traffic = 0
        capacity_area_list = []
        tot_capacity = 0
        critical_loads = np.zeros(len(self.areas))
        critical_loads_ol = np.zeros(len(self.areas))
        for i, area in enumerate(self.areas):
            #logging.info(f" ---------- Area {i} : {area.area_name} ----------")
            old_load_report , new_load_report,ideal_load_report,traffic_old, traffic_new, traffic_ideal,traffic_sum_area, new_traffic_sum_area, ideal_traffic_per_area , capacity_area = area.FinalReport()
            
            new_traffics_sum.append(new_traffic_sum_area)
            traffic_sum.append(traffic_sum_area)
            if capacity_area != 0:
                old_load_dist[i] = traffic_sum_area / capacity_area
                new_load_dist[i] = new_traffic_sum_area / capacity_area
                ideal_load_dist[i] = ideal_traffic_per_area / capacity_area
                

            total_new_traffic+= new_traffic_sum_area
            total_old_traffic+= traffic_sum_area
            total_ideal_traffic+= ideal_traffic_per_area
            #print(area.area_name, i)
            #print(area.critical_load, area.critical_load_ol)
            critical_loads[i] = area.critical_load
            critical_loads_ol[i] = area.critical_load_ol
            capacity_area_list.append(capacity_area)
            tot_capacity += capacity_area
            
            traffic_old_list.append(traffic_old)
            traffic_new_list.append(traffic_new)

            old_traffic_load_per_area.append(old_load_report)
            new_traffic_load_per_area.append(new_load_report)
            ideal_traffic_load_per_area.append(ideal_load_report)
            #logging.info(f" Load Reduction {(traffic_sum_area - new_traffic_sum_area)/capacity_area:e}")
            area.data_rate.to_csv(os.path.join(path, f"{area.area_name}_data_rate.csv"))
            if plot_bs : 
                area.plot()
            logging.info("\n")
        #print(tot_capacity, total_old_traffic)
        logging.info(f"Ideal Load Reduction: {(total_old_traffic - total_ideal_traffic)/tot_capacity}) ")
        logging.info(f"Total Load Reduction: {(total_old_traffic - total_new_traffic)/tot_capacity}")
        print(f"Ideal Load Reduction: {(total_old_traffic - total_ideal_traffic)/tot_capacity}) ")
        print(f"Total Load Reduction: {(total_old_traffic - total_new_traffic)/tot_capacity}")
        #logging.info(f"chosen areas: {self.areas_chosen_list}")    
        logging.info(f"Mean chosen areas: {np.mean(self.n_areas_history)}, std chosen areas: {np.std(self.areas_chosen_list)}")
        logging.info(f"average choosen area : {self.n_areas_history}")    
        print(np.sum(critical_loads)/ np.sum(critical_loads_ol))
        print(np.sum(critical_loads), np.sum(critical_loads_ol))
        self.haps_positions_df.to_csv(os.path.join(path,"haps_positions.csv"))
        self.haps_beams_load.to_csv(os.path.join(path,".haps_beams_load.csv"))
        self.load_history.to_csv(os.path.join(path,"load_history.csv"))
        self.haps_capacity_per_beam.to_csv(os.path.join(path,"haps_capacity_per_beam.csv"))
        self.Area_selection_df.to_csv(os.path.join(path,"area_selection.csv"))
        np.save(os.path.join(path,"reward_error_list_random.npy"),self.reward_error_list)
        #np.save("results/beam_loads_sum.npy",self.haps_beams_loadc)
        #np.save("results/haps_loads_sum.npy",self.load_history)
        #np.save("results/traffic_new_list.npy",np.array(traffic_new_list))
        #np.save("results/traffic_old_list.npy",np.array(traffic_old_list))
        #np.save("results/critical_loads_3.npy",critical_loads)   
        #np.save("results/critical_loads_ol_3.npy",critical_loads_ol)
        #self.los_df.to_csv("./results/los_angles.csv")   
        #self.haps_positions_df.to_csv("./results/haps_positions.csv")
        #np.save("results/capacity_per_area.npy",np.array(capacity_area_list))
        return  old_load_dist, new_load_dist, ideal_load_dist, old_traffic_load_per_area, new_traffic_load_per_area, ideal_traffic_load_per_area, (total_old_traffic - total_new_traffic)/tot_capacity, (total_old_traffic - total_ideal_traffic)/tot_capacity


    def update(self, time_int : int)-> None:
        time = self.time[time_int]
        self.update_areas_ps_buffer(time_int)
        if 23 >time.hour >= 8: 
            self.hourly_update(time)
        if time.hour == 23:

            for area in self.areas:
                area.hps_buffer.clear()


    def hourly_update(self, time)-> None:
        def get_part_of_day(time):
            hour = time.hour
            if 8 <= hour < 16:
                return 'Day'
            elif 16 <= hour <= 23:
                return 'Night'
        
        part_of_day = get_part_of_day(time)
        day = time.day_name()
        i = 0
        if self.algo == "exp3": 
            z_score_areas = self.select_values_adaptive(self.Values_df.loc[day + "-" + part_of_day ], z_threshold= 0)
            #print(z_score_areas)
        day_part_of_day = day + "-" + part_of_day   
        Value_area_array = self.Values_df.loc[day_part_of_day, :].values
        #print(Value_area_array)
        Value_array_min_max = 0.8*(Value_area_array - np.min(Value_area_array))/(np.max(Value_area_array) - np.min(Value_area_array)) if np.max(Value_area_array) != np.min(Value_area_array) else np.ones(len(Value_area_array))*(-0.25)
        logging.info(f"Area min max : {Value_array_min_max}")
        
        for area in self.areas:
            day_part_of_day = day + "-" + part_of_day   
            hourly_ops = area.get_hourly_ps()
            self.hourly_ps_buffer[i] = hourly_ops
            Value_area = self.Values_df.loc[day_part_of_day, area.area_name]
            #if part_of_day == "Night":
            #    previous_influence
            if self.algo == "exp3":
                x = 0
                if  i in z_score_areas:
                #    logging.info(time)
                #    logging.info(f"{i} , {z_score_areas}")
                #    #self.Values_df.loc[day_part_of_day, area.area_name] = self.gamma*Value_area
                #    self.Values_df.loc[day_part_of_day, area.area_name] = 0.95*Value_area
                #    i += 1
                #    continue
                    x = 0.1
                alpha = np.clip(1/ (1 + Value_array_min_max[i]), 0.55,0.8)
                    
                self.Values_df.loc[day_part_of_day, area.area_name] = \
                self.gamma*(hourly_ops + alpha*(Value_area + hourly_ops) - Value_area)  +  Value_area  
                #0.95 *(hourly_ops - Value_area) + Value_area 
                
            #hourly_ops + Value_area
            if self.algo == "ucb":  
            #    if i in self.ChosenAreas: 
                    self.N.loc[day_part_of_day, area.area_name] += 1
            #        area.hps_window[day_part_of_day].append(hourly_ops)
            elif self.algo == "greedy":
                #if hourly_ops == 0:
                #    
                #    self.Values_df.loc[day_part_of_day, area.area_name] = 0
                #    continue
                self.Values_df.loc[day_part_of_day, area.area_name] = hourly_ops 
            elif self.algo == "exp":
                
                #    new_hps = hourly_ops/ self.probs_exp3[i]
                #    Value_area = self.Values_df.loc[day_part_of_day, area.area_name]
                #    self.Values_df.loc[day_part_of_day, area.area_name]  = Value_area * np.exp(self.gamma*new_hps/len(self.areas))
                    self.N.loc[day_part_of_day, area.area_name] += 1
                    N = self.N.loc[day_part_of_day, area.area_name]
                    # Update mean
                    self.means.loc[day_part_of_day, area.area_name] = ((self.means.loc[day_part_of_day, area.area_name] * (N - 1)) + hourly_ops) / N
                    # Update variance using Welford's method
                    self.variances.loc[day_part_of_day, area.area_name] = ((self.variances.loc[day_part_of_day, area.area_name] * (N - 1)) + 
                                                (hourly_ops - self.means.loc[day_part_of_day, area.area_name]) * (hourly_ops - self.means.loc[day_part_of_day, area.area_name])) / N

            elif self.algo == "epsilon-greedy": 
                
                if i in self.ChosenAreas : 
                    self.N.loc[day_part_of_day, area.area_name] += 1
                    N = self.N.loc[day_part_of_day, area.area_name]
                    #print(i,N)
                    #print(i , self.ChosenAreas)
                    self.Values_df.loc[day_part_of_day, area.area_name]  = Value_area + 1 / N*(hourly_ops - Value_area)
            i+=1
            #hourly_ops + self.alpha*(Value_area + hourly_ops)
            #logging.info(f"Area {area.area_name} : {Value_area}")
            #logging.info(f"Hourly ops: {hourly_ops}")
            #logging.info(f" update : {hourly_ops + self.alpha*(Value_area+hourly_ops)}")
    
    def reward_error(self, time_int : int)-> None: 
       
        if time_int > self.getTrainTime():
            ps_ideal = sorted(self.hourly_ps_buffer, reverse = True)
            print(ps_ideal)
            ps_ideal_chosen = np.array(ps_ideal)[:len(self.ChosenAreas)] 
            ps_real = self.hourly_ps_buffer[self.ChosenAreas]
            error = np.sum(ps_ideal_chosen)- np.sum(ps_real)
            self.reward_error_list.append(error)
            

    def objective_function(self, haps_position, area_points, ChosenAreas,areas_boundaries ,method = "PSO", plot = False,theta = 0.75 )-> None:

        # Convert to Cartesian coordinates
        ## normalize the weights
        z_value = 0  # You can change this to any value or logic you want for the z-coordinate.
        area_points_with_z = [(x, y, z_value) for x, y in area_points]
        
        area_parameters = []
        choosen_areas_types  = self.area_type_array[ChosenAreas]
        choosen_indoor_users_percentage = self.indoor_users_array[ChosenAreas]
        choosen_traditional_building_percentage = self.traditional_building_percentage_array[ChosenAreas]
        #print(choosen_areas_types)
        #z_value = 0  # You can change this to any value or logic you want for the z-coordinate.
        idx = 0
        map_boundary =   self.haps_boundaries
        #haps_position = ( centroid_x, centroid_y, 20000)
        for coordinate,area_type, indoor_percentage, traditional_building  in zip(area_points_with_z, choosen_areas_types, choosen_indoor_users_percentage, choosen_traditional_building_percentage):
            area_dict = dict()
            coordinate = (coordinate[0]* 10**6, coordinate[1] * 10**6, 0)
            area_dict["area_coordinates"] = coordinate
            area_dict["area_name"] = self.area_names[ChosenAreas[idx]]
            area_dict["area_type"] = area_type
            area_dict["indoor_percentage"] = indoor_percentage
            area_dict["traditional_building_percentage"] = traditional_building
            if method != "static": 
                area_dict["N_sim"] = 500
            else : 
                area_dict["N_sim"] = 500
            area_dict["boundary"] = areas_boundaries[idx]
            #print(coordinate)
            
            receiver_point = (coordinate[0], coordinate[1], 0)  # Example receiver point with z = 0
            centroid_point = (haps_position[0]*10**6, haps_position[1]*10**6, 20000)  # Centroid with z = 20000
            #print(receiver_point, centroid_point)   
            # Compute the horizontal distance between the receiver and the centroid
            horizontal_distance = np.sqrt((centroid_point[0] - receiver_point[0])**2 + (centroid_point[1] - receiver_point[1])**2)

            # Compute the height difference
            height_difference = centroid_point[2] - receiver_point[2]

            # Compute the elevation angle in radians
            elevation_angle_rad = np.arctan2(height_difference, horizontal_distance)

            # Convert the elevation angle to degrees
            elevation_angle_deg = np.degrees(elevation_angle_rad)
            #print(area_dict["area_name"])
            #print(elevation_angle_deg)
            area_dict["elevation_angle_deg"] = elevation_angle_deg
            closest_data = min(self.haps_sband_parameters.data, key=lambda d: abs(d.elevation - elevation_angle_deg))
            #print(closest_data)
            area_dict["SBAND_data"] = closest_data
            area_parameters.append(area_dict)
            idx += 1
        capacities= self.HAPS_Channel_model.generate_capacities(area_parameters,centroid_point, self.n_areas, 16,plot=  plot)
        print(capacities)
        
        #print(np.mean(capacities)/ np.min(capacities)* np.std(capacities) , np.sum(capacities))
        #print(entropy(capacities))
        #self.final_capacities = deepcopy(capacities)
        minimum_capacity = np.min(capacities)
        min_constraint = 20
        penalty = 1000 if minimum_capacity < min_constraint else 0
        #capacities_values = area_values * capacities
        objective = np.sum(capacities) - theta * np.std(capacities)* np.mean(capacities)/np.min(capacities) - penalty
        #distance = np.sqrt((haps_position[0]*10**6 - haps_previous_pos[0]*10**6)**2 + (haps_position[1]*10**6 - haps_previous_pos[1]*10**6)**2)
        if method == "PSO":
            if -objective < -self.pso_best_objective :
                self.pso_best_objective = deepcopy(objective)
                self.pso_best_position = deepcopy(haps_position)
                self.final_capacities_pso = deepcopy(capacities)
        elif method == "scipy": 
            if -objective < -self.pso_best_objective:
                self.scipy_objective = deepcopy(objective)
                self.haps_position_buffer = deepcopy(haps_position)
                self.final_capacities_scipy = deepcopy(capacities)
                self.scipy_flag = True
        else : 
            #self.haps_position = haps_position
            self.final_capacities = deepcopy(capacities)
            self.objective = -deepcopy(objective)
    
        #if abs(self.buffer_objective) < abs(objective):
        #     
        #    self.final_objective_value = deepcopy(objective)
        #    self.haps_position_buffer = deepcopy(haps_position)
        #    self.buffer_objective = deepcopy(objective)
        return -objective

        #self.los_df.loc[los_angle_df.Centroid_Name, time] = los_angle_df.Angle_Deg.values.astype(float)
    def find_haps_position(self, choosen_area_coordinates : List,ChosenAreas : List, time : int, mini : bool = True)-> None: 
        area_points = choosen_area_coordinates.centroid.apply(lambda p: (p.x, p.y)).tolist()
        
        #print(area_points)
        x_coords = [point[0] for point in area_points]
        y_coords = [point[1] for point in area_points]
        #self.buffer_objective = 0 
        #print("HAPS POSITION ",self.haps_position)
        #centroid_x = np.mean(x_coords)
        #centroid_y = np.mean(y_coords)
        #self.haps_position = (centroid_x/ 10**6, centroid_y/10**6)
        self.haps_position = [self.haps_position[0]/10**6, self.haps_position[1]/10**6]
        area_points = [(x/10**6, y/10**6) for x, y in area_points]
        bounds = [(self.haps_boundaries[0]/10**6, self.haps_boundaries[2]/10**6), (self.haps_boundaries[1]/10**6, self.haps_boundaries[3]/10**6)]
        areas_boundaries = choosen_area_coordinates.geometry.tolist()
        #print(self.haps_position)
        constraints = [{
            'type': 'ineq',
            'fun': lambda haps_position: self.circular_constraint(haps_position[:2], self.haps_position[:2])
        }]
        if mini : 
            #old_objective_value = self.objective_function(self.haps_position, area_points)
            #self.buffer_objective = 0 
            print("haps position",self.haps_position)
            #old_position_haps_capacities = deepcopy(self.final_capacities)
            result = minimize(
            self.objective_function, self.haps_position, args=( area_points,),
            method='COBYQA', options={'maxiter': 1000}, bounds=bounds, constraints=constraints,)
            #print("Minimum objective value", result.fun)    
            #new_objective_value = result.fun
            #if abs(new_objective_value) > 1.1 * abs(old_objective_value)  : 
            self.haps_position = [result.x[0]*10**6, result.x[1]*10**6]
            #else : 
                #self.final_capacities = old_position_haps_capacities
                #self.haps_position = [self.haps_position[0]*10**6, self.haps_position[1]*10**6]
        #print("SOLUTIONS HAPS POSITION ",result.x , "bounds", bounds)
        else : 
            #logging.info(los_angle_df)
            #_ = self.objective_function(self.haps_position, area_points, ChosenAreas, "static")
            _ =self.objective_function(self.haps_position, area_points, ChosenAreas, areas_boundaries,"static", plot = False)
            self.haps_position = [self.haps_position[0]*10**6, self.haps_position[1]*10**6]
        #print("HAPS new position ",self.haps_position)
        #self.haps_positions_df.iloc[time, 0] = self.haps_position[0]
        #self.haps_positions_df.iloc[time, 1] = self.haps_position[1]
        #logging.info(f"Haps capacity: {self.final_capacities}") 
        ##
        ## print(self.haps_positions_df)
        #final_capacities_in_bits_per_hour = np.array([c * 3600 * 10**6 for c in self.final_capacities])
        ##print("haps traffic",self.haps_traffic)
        #print("final capacities Mbit/s", self.final_capacities)
        #print("final capacities bits per hour",final_capacities_in_bits_per_hour)
        #print("traffic", self.haps_traffic)
        ##print(self.haps_traffic / final_capacities_in_bits_per_hour)
        ##print(self.haps_beams_load, len(self.haps_traffic/final_capacities_in_bits_per_hour))
        #beams_load_array = self.haps_traffic/final_capacities_in_bits_per_hour
        #if len(beams_load_array[beams_load_array > 1]) > 0:
        #    logging.info("Beams load greater than 1")
        #    index = np.where(beams_load_array > 1)
        #    logging.info(index)
        #    logging.info(f"haps capacities {final_capacities_in_bits_per_hour}" )
        #self.haps_capacity_per_beam.iloc[:,time] = final_capacities_in_bits_per_hour/(10**6*3600)
        #self.haps_beams_load.iloc[:,time] = self.haps_traffic/final_capacities_in_bits_per_hour
        #self.load_history.iloc[0, time] =  np.sum(self.haps_traffic)/np.sum(final_capacities_in_bits_per_hour)
        ##print(self.haps_beams_load)
        #_ =self.objective_function(haps_position, area_points, ChosenAreas, areas_boundaries,"static", plot = True)



    def position_controller(
        self, 
        choosen_area_coordinates: List, 
        values_array: np.ndarray, 
        ChosenAreas: List, 
        mini: bool
    ) -> List:
        """
        Optimize the HAPS position using a hybrid PSO-SciPy approach.
        
        Parameters:
        - choosen_area_coordinates: List of coordinates for selected areas.
        - values_array: Array of current estimated values for each area.
        - ChosenAreas: List of indices for selected areas.
        - mini: Boolean flag indicating whether to run the minimization process.

        Returns:
        - ChosenAreas: Updated list of selected areas.
        """
        # Normalize initial HAPS position and area points
        area_points = choosen_area_coordinates.centroid.apply(lambda p: (p.x, p.y)).tolist()
        self.haps_position = [self.haps_position[0] / 10**6, self.haps_position[1] / 10**6]
        area_points = [(x / 10**6, y / 10**6) for x, y in area_points]
        
        areas_boundaries = choosen_area_coordinates.geometry.tolist()
        print(areas_boundaries, len(areas_boundaries))
        # Define bounds
        bounds = [
            (self.haps_boundaries[0] / 10**6, self.haps_boundaries[2] / 10**6),
            (self.haps_boundaries[1] / 10**6, self.haps_boundaries[3] / 10**6)
        ]
        print(bounds)
        # Step 1: Global Optimization with PSO
        constraints = [{
            'type': 'ineq',
            'fun': lambda haps_position: self.circular_constraint(haps_position[:2], self.haps_position[:2])
        }]
        pso_constraint = lambda haps_position: constraints[0]['fun'](haps_position)
        def combined_constraint(haps_position, *args):
            """
            Enforce:
            - Circular boundary around the initial HAPS position.
            - Minimum capacity constraint for all areas.
            """
              # Extract additional arguments
            #capacities = self.calculate_capacities(haps_position, area_points, chosen_areas)  # Calculate capacities
            # Circular boundary constraint
            center = self.haps_position[:2]
            radius = self.Dmax*1_000   # Define radius in normalized units
            distance = np.sqrt((haps_position[0]*10**6 - center[0]*10**6)**2 + (haps_position[1]*10**6 - center[1]*10**6)**2)
            circular_constraint = radius - distance  # Positive if inside the circle

            # Minimum capacity constraint
            #capacity_constraints = [cap - min_capacity for cap in capacities]  # Positive if cap >= min_capacity

            # Combine constraints: all must be satisfied
            return circular_constraint
        # Penalized objective function for PSO
   
   
        def bounding_box_from_circle(center, radius):
            """
            Calculate the minimum bounding box for a full circle.

            Parameters:
                center (tuple): Coordinates of the circle's center (x, y).
                radius (float): Radius of the circle.

            Returns:
                dict: Bounding box with min_x, max_x, min_y, max_y.
            """
            x, y = center
            x = x * 10**6
            y = y * 10**6
        
            # Compute bounding box directly from the circle's radius
            min_x = x - radius
            max_x = x + radius
            min_y = y - radius
            max_y = y + radius

            
            return[
                 [min_x / 10**6,
                 max_x/ 10**6],
                 [min_y/ 10**6,
                 max_y/ 10**6]
            ]
        haps_previous_position = deepcopy(self.haps_position)   
        # Step 3: Check Capacity Constraints and Adjust Areas
        while mini:
            self.pso_best_objective = -np.inf
            print("Starting PSO...")
            self.scipy_flag = False
            print(ChosenAreas)
            #pso_result, pso_score = pso(self.compute_los_objective, [b[0] for b in bounds],
            #                            [b[1] for b in bounds],  # Lower bounds
            #  swarmsize = 100, maxiter = 50, args=(area_points, ChosenAreas), debug=False, minstep=1e-2,  )
            #print("PSO Complete. Best Position:", pso_result, "Score:", -pso_score)
            ##_= self.compute_los_objective(pso_result, area_points, ChosenAreas)
            #
            ##for idx in range(15): 
            ##    print("-------------------------------")
            ##    result= self.compute_los_objective([pso_result[0], pso_result[1]+(idx+1)*1e-3], area_points, ChosenAreas)
            ##    print("----------new position", pso_result[0], pso_result[1]+(idx+1)*1e-3)
            ##    print("----------",-result,"--------")
            #distance_bary_old_point = np.sqrt((haps_previous_position[0]*10**6 - pso_result[0]*10**6)**2 + (haps_previous_position[1]*10**6 - pso_result[1]*10**6)**2)
            #self.bary_center = (pso_result[0], pso_result[1])
            #print("ditstance 1", distance_bary_old_point)
            #logging.info(f"Distance between old and bary: {distance_bary_old_point}")
            ##if distance_bary_old_point < 6000: 
            constraint = combined_constraint
            bounds = bounding_box_from_circle(haps_previous_position, self.Dmax*1_000)
            #else : 
            #    constraint = pie_slice_constraint_facing_A
            #    bounds = bounding_box_from_points_and_pie(self.bary_center, haps_previous_position, 6_000)
        # Start PSO with combined constraints
            pso_result, pso_score = pso( 
            self.objective_function,
            [b[0] for b in bounds],  # Lower bounds
            [b[1] for b in bounds],  # Upper bounds
            swarmsize=30,            # Number of particles
            maxiter=25,              # Maximum iterations
            args=(area_points, ChosenAreas,areas_boundaries,"PSO"),  # Pass additional arguments to the objective function
            minstep=1e-4,            # Minimum step size
            debug=True,
            f_ieqcons=constraint # Use the combined constraint

              # Pass min_capacity to the constraint function
        )       
            logging.info(f"PSO Complete. Best Position: {pso_result} Score: {-pso_score}")
            
           
            #print("PSO Capacities", self.final_capacities_pso)
            # Step 2: Local Refinement with SciPy
            
            #pso_capacity_solution = deepcopy(self.final_capacities)
        
        
            #self.plot_pie(pso_result, haps_previous_position, 8_000)
        
        
            #self.final_capacities = deepcopy(self.final_capacities_pso)
            _ =self.objective_function(self.haps_position, area_points, ChosenAreas, areas_boundaries,"static")
            print("ditstance 2", np.sqrt((haps_previous_position[0]*10**6 - pso_result[0]*10**6)**2 + (haps_previous_position[1]*10**6 - pso_result[1]*10**6)**2))
            
            if self.objective > 0.95 * pso_score :
                print("Change position ....................................")
                logging.info("Change position ....................................")
                self.haps_position = [pso_result[0]*10**6, pso_result[1]*10**6]
                haps_position = [ self.haps_position[0] / 10**6, self.haps_position[1] / 10**6]
                _ = self.objective_function(haps_position, area_points, ChosenAreas, areas_boundaries,"static", plot = False)
                
                #self.haps_position = [self.pso_best_position[0]*10**6, self.pso_best_position[1]*10**6]
            #if all( c >= self.min_capacity  for c in self.final_capacities):
            #print("distance between old and new position", np.sqrt((haps_previous_position[0]*10**6 - self.pso_best_position[0]*10**6)**2 + (haps_previous_position[1]*10**6 - self.pso_best_position[1]*10**6)**2))
            
            else : 
                self.haps_position = [self.haps_position[0] * 10**6, self.haps_position[1] * 10**6]
            #area_points = [(x / 10**6, y / 10**6) for x, y in area_points]
            logging.info(f"Final HAPS Position: {self.haps_position}")
            logging.info(f"Final Capacities: {self.final_capacities}")
            print(f"All areas meet minimum capacity ({self.min_capacity} Mbps).")
            print("Final Capacities", self.final_capacities)
            #self.haps_position = [result.x[0] * 10**6, result.x[1] * 10**6]
            break
            #elif len(ChosenAreas) == 4:
            #    if self.scipy_flag : 
            #        self.haps_position = [self.haps_position_buffer[0]*10**6, self.haps_position_buffer[1]*10**6]
            #        #self.final_capacities = deepcopy(self.final_capacities_scipy)
            #    else :
            #        #self.final_capacities = deepcopy(self.final_capacities_pso)
            #        self.haps_position = [self.pso_best_position[0]*10**6, self.pso_best_position[1]*10**6]
            #    print("Minimum number of areas reached (4).")
            #    haps_position = [ self.haps_position[0] / 10**6, self.haps_position[1] / 10**6]
            #    #area_points = [(x / 10**6, y / 10**6) for x, y in area_points]
            #    _ =self.objective_function(haps_position, area_points, ChosenAreas, areas_boundaries,"static", plot = True)
            #    #self.haps_position = [result.x[0] * 10**6, result.x[1] * 10**6]
            #    break
            #else:
            #    # Remove the area with the lowest value
            #    area_points = [(x * 10**6, y * 10**6) for x, y in area_points]
            #    #dbscan = DBSCAN(eps=10000 , min_samples=2)
            #    print(area_points)
            #    #dbscan_clusters = dbscan.fit_predict(area_points)
            #    print("values", values_array)
            #    #new_values = 0.5*values_array + self.final_capacities
            #    #print(dbscan_clusters)
            #    #new_values = 0.5*values_array + self.final_capacities
            #    #areas_to_remove_indices = np.argwhere(dbscan_clusters == -1).flatten()
            #    #if len(areas_to_remove_indices) == 0:
            #    area_to_remove = np.argmin(values_array)
            #    #print("new values", new_values)
            #    #else : 
            #    #    area_to_remove = np.argmin(np.array(self.final_capacities)[areas_to_remove_indices])
            #    #print("new values", new_values)
            #    #area_to_remove = np.argmin(np.array(self.final_capacities)[areas_to_remove_indices])
            #    print(area_to_remove)
            #    print("remove",area_to_remove)
            #    print(f"Removing area: {area_to_remove}")
#
            #    # Update the list of chosen areas
            #    choosen_area_coordinates = choosen_area_coordinates.drop(
            #        choosen_area_coordinates.index[area_to_remove]
            #    )
            #    area_points = choosen_area_coordinates.centroid.apply(lambda p: (p.x, p.y)).tolist()
            #    area_points = [(x / 10**6, y / 10**6) for x, y in area_points]
            #    values_array = np.delete(values_array, area_to_remove)
            #    ChosenAreas = np.delete(ChosenAreas, area_to_remove)
            #    areas_boundaries = choosen_area_coordinates.geometry.tolist()
            #self.n_areas -= 1
        # Store the history of selected areas and HAPS positions
        #self.n_areas_history.append(self.n_areas)
        return ChosenAreas


    def position_controller_local(self, choosen_area_coordinates : List, values_array : np.ndarray, ChosenAreas : List, mini : bool)-> None: 
        area_points = choosen_area_coordinates.geometry.apply(lambda p: (p.x, p.y)).tolist()
        self.haps_position = [self.haps_position[0]/10**6, self.haps_position[1]/10**6]
        area_points = [(x/10**6, y/10**6) for x, y in area_points]
        bounds = [(self.haps_boundaries[0]/10**6, self.haps_boundaries[2]/10**6), (self.haps_boundaries[1]/10**6, self.haps_boundaries[3]/10**6)]
        constraints = [{
            'type': 'ineq',
            'fun': lambda haps_position: self.circular_constraint(haps_position[:2], self.haps_position[:2])
        }]
        if not mini : 
            _ = self.objective_function(self.haps_position, area_points)
            self.haps_position = [self.haps_position[0]*10**6, self.haps_position[1]*10**6]
            
        while True and mini: 
            print(choosen_area_coordinates)
            result = minimize(
            self.objective_function, self.haps_position, args=( area_points,ChosenAreas),
            method='COBYQA', options={'maxiter': 1000}, bounds=bounds, constraints=constraints,)
            if all( c >= self.min_capacity for c in self.final_capacities):
                print("Number of areas", self.n_areas)
                self.haps_position = [result.x[0]*10**6, result.x[1]*10**6]
                break
            elif len(ChosenAreas) == 4:
                #self.n_areas += 1
                self.haps_positions_df = [result.x[0]*10**6, result.x[1]*10**6]
                break 
            else : 
                dbscan = DBSCAN(eps=10000, min_samples=2)
                dbscan_clusters = dbscan.fit_predict(area_points)
                print("values", values_array)
                #new_values = 0.5*values_array + self.final_capacities
                areas_to_remove_indices = np.argwhere(dbscan_clusters == -1).flatten()
                if len(areas_to_remove_indices) == 0:
                    area_to_remove = np.argmin(values_array)
                #print("new values", new_values)
                else : 
                    area_to_remove = np.argmin(values_array[areas_to_remove_indices])
                print("remove",area_to_remove)
                choosen_area_coordinates = choosen_area_coordinates.drop(choosen_area_coordinates.index[area_to_remove])
                area_points = choosen_area_coordinates.geometry.apply(lambda p: (p.x, p.y)).tolist()
                area_points = [(x/10**6, y/10**6) for x, y in area_points]
                values_array = np.delete(values_array, area_to_remove)
                ChosenAreas = np.delete(ChosenAreas, area_to_remove)
            self.n_areas -= 1
        self.n_areas_history.append(self.n_areas)
        return ChosenAreas    
    
    def circular_constraint(self, haps_position, initial_guess):    
        """
        Constraint to keep HAPS position within a circular boundary.
        
        Parameters:
        - haps_position: Current HAPS position (x, y) to check.
        - initial_guess: Initial guess (center of the circle).
        
        Returns:
        - Distance constraint value: Should be non-negative if within the boundary.
        """
        # Calculate the Euclidean distance from initial guess
        distance = np.sqrt((haps_position[0] - initial_guess[0])**2 + (haps_position[1] - initial_guess[1])**2)
        radius = 1_000 / 10**6  # 3.5 km
        return radius - distance  # Non-negative if within the circle

    def position_report(self, time) : 
        self.haps_positions_df.iloc[time, 0] = self.haps_position[0]
        self.haps_positions_df.iloc[time, 1] = self.haps_position[1]
        #print(self.haps_positions_df)
        final_capacities_in_bits_per_hour = np.array([c * 3600 * 10**6 for c in self.final_capacities])
        #print("haps traffic",self.haps_traffic)
        print("final capacities Mbit/s", self.final_capacities)
        print("final capacities bits per hour",final_capacities_in_bits_per_hour)
        print("traffic", self.haps_traffic)
        #print(self.haps_traffic / final_capacities_in_bits_per_hour)
        #print(self.haps_beams_load, len(self.haps_traffic/final_capacities_in_bits_per_hour))
        beams_load_array = self.haps_traffic/final_capacities_in_bits_per_hour
        print(beams_load_array)
        if len(beams_load_array[beams_load_array > 1]) > 0:
            logging.info("Beams load greater than 1")
            index = np.where(beams_load_array > 1)
            logging.info(index)
        logging.info(f"haps capacities bits per hour {final_capacities_in_bits_per_hour}" )
        logging.info(f"haps capacities Mbit/s {self.final_capacities}" )
        
        self.haps_capacity_per_beam.iloc[:self.n_areas,time] = final_capacities_in_bits_per_hour/(10**6*3600)
        self.haps_beams_load.iloc[:self.n_areas,time] = self.haps_traffic/final_capacities_in_bits_per_hour
        self.load_history.iloc[0, time] =  np.sum(self.haps_traffic)/np.sum(final_capacities_in_bits_per_hour)
        for idx , area in enumerate(self.areas):
            if idx in self.ChosenAreas:
                beam_arg =np.where(self.ChosenAreas == idx)[0][0]          
                area.data_rate.iloc[time] = self.final_capacities[beam_arg]
                print(area.data_rate.iloc[time] , area.area_name)
    def find_haps_positio_static(self, time, flag): 
        
        #print(area_points)
        #x_coords = [point[0] for point in area_points]
        #y_coords = [point[1] for point in area_points]
      
        #print("HAPS POSITION ",self.haps_position)
        #centroid_x = np.mean(x_coords)
        #centroid_y = np.mean(y_coords)
        #self.haps_position = (centroid_x/ 10**6, centroid_y/10**6)
        self.haps_position = [self.haps_position[0]/10**6, self.haps_position[1]/10**6]
        #area_points = [(x/10**6, y/10**6) for x, y in area_points]
        bounds = [(self.haps_boundaries[0]/10**6, self.haps_boundaries[2]/10**6), (self.haps_boundaries[1]/10**6, self.haps_boundaries[3]/10**6)]
        
        #print(self.haps_position)
                
        pso_result, pso_score = pso(
            self.objective_function_static,
            [b[0] for b in bounds],  # Lower bounds
            [b[1] for b in bounds],  # Upper bounds
            swarmsize=25,            # Number of particles
            maxiter=25,   
             args=(flag,), # Maximum iterations
          # Pass additional arguments to the objective function
            minstep=1e-4,            # Minimum step size
            debug=True,
          # Use the combined constraint
              # Pass min_capacity to the constraint function
        )
        #result = minimize(
        #        self.objective_function_static,
        #        x0=self.haps_position,  # Initial guess from PSO
        #        bounds=bounds,
        #        method='SLSQP',  # Use SLSQP if you have constraints
        #        options={'maxiter': 10_000},
        #        
        #    )
            
        #print("SOLUTIONS HAPS POSITION ",result.x , "bounds", bounds)
        self.haps_position = [pso_result[0]*10**6, pso_result[1]*10**6]
        #print("HAPS new position ",self.haps_position)
            #logging.info(los_angle_df)
        #_ = self.objective_function(self.haps_position, area_points)
        #self.haps_positions_df.iloc[time, 0] = self.haps_position[0]
        #self.haps_positions_df.iloc[time, 1] = self.haps_position[1]
        print("haps position",self.haps_position)
        print("final position", pso_result[0]*10**6, pso_result[1]*10**6)
        logging.info(f"HAPS POSITION {self.haps_position}")
        #np.save("results/haps_position.npy",self.haps_position)
        #print(self.haps_positions_df)
        #final_capacities_in_bits_per_hour = np.array([c * 3600 * 10**6 for c in self.final_capacities])
        #print("haps traffic",self.haps_traffic)
        #print("final capacities",final_capacities_in_bits_per_hour)
        #print(self.haps_traffic / final_capacities_in_bits_per_hour)
        #print(self.haps_beams_load, len(self.haps_traffic/final_capacities_in_bits_per_hour))
        #beams_load_array = self.haps_traffic/final_capacities_in_bits_per_hour
        #if len(beams_load_array[beams_load_array > 1]) > 0:
        #    logging.info("Beams load greater than 1")
        #    index re= np.where(beams_load_array > 1)
        #    logging.info(index)
        #    logging.info(f"haps capacities {final_capacities_in_bits_per_hour}" )
        #self.haps_capacity_per_beam.iloc[:,time] = final_capacities_in_bits_per_hour/(10**6*3600)
        #self.haps_beams_load.iloc[:,time] = self.haps_traffic/final_capacities_in_bits_per_hour
        #self.load_history.iloc[0, time] =  np.sum(self.haps_traffic)/np.sum(final_capacities_in_bits_per_hour)
        
    
    def objective_function_static(self, haps_position, flag)-> None:

        # Convert to Cartesian coordinates
        ## normalize the weights
        final_objective = 0
        #combs=[14, 2, 1, 0, 3, 15, 11, 6, 9, 8]
        if flag : 
            combs = [ 0,  1,  2,  3,  6,  8, 11, 12, 14, 15]
        else : 
            combs = [ 0,  1,  3,  5,  6,  8,  9, 11, 14, 15]
        for ChosenAreas in combinations(combs, 8):
            #print(ChosenAreas)
            ChosenAreas = list(ChosenAreas)
            #print(self.area_centorid)
            choosen_area_coordinates = self.area_centorid.iloc[ChosenAreas]
            area_points = choosen_area_coordinates.centroid.apply(lambda p: (p.x, p.y)).tolist()

            area_points = [(x/10**6, y/10**6) for x, y in area_points]
            z_value = 0  # You can change this to any value or logic you want for the z-coordinate.
            area_points_with_z = [(x, y, z_value) for x, y in area_points]
            
            area_parameters = []
            choosen_areas_types  = self.area_type_array[ChosenAreas]
            choosen_indoor_users_percentage = self.indoor_users_array[ChosenAreas]
            choosen_traditional_building_percentage = self.traditional_building_percentage_array[ChosenAreas]
            #print(choosen_areas_types)
            #z_value = 0  # You can change this to any value or logic you want for the z-coordinate.
        
            #haps_position = ( centroid_x, centroid_y, 20000)
            for coordinate,area_type, indoor_percentage, traditional_building  in zip(area_points_with_z, choosen_areas_types, choosen_indoor_users_percentage, choosen_traditional_building_percentage):
                area_dict = dict()
                coordinate = (coordinate[0]* 10**6, coordinate[1] * 10**6, 0)
                area_dict["area_coordinates"] = coordinate
                area_dict["area_type"] = area_type
                area_dict["indoor_percentage"] = indoor_percentage
                area_dict["traditional_building_percentage"] = traditional_building
                area_dict["N_sim"] = 100
                
                receiver_point = (coordinate[0], coordinate[1], 0)  # Example receiver point with z = 0
                centroid_point = (haps_position[0]*10**6, haps_position[1]*10**6, 20000)  # Centroid with z = 20000
                #print(receiver_point, centroid_point)   
                # Compute the horizontal distance between the receiver and the centroid
                horizontal_distance = np.sqrt((centroid_point[0] - receiver_point[0])**2 + (centroid_point[1] - receiver_point[1])**2)

                # Compute the height difference
                height_difference = centroid_point[2] - receiver_point[2]

                # Compute the elevation angle in radians
                elevation_angle_rad = np.arctan2(height_difference, horizontal_distance)

                # Convert the elevation angle to degrees
                elevation_angle_deg = np.degrees(elevation_angle_rad)
                #print(elevation_angle_deg)
                
                area_dict["elevation_angle_deg"] = elevation_angle_deg
                closest_data = min(self.haps_sband_parameters.data, key=lambda d: abs(d.elevation - elevation_angle_deg))
                #print(closest_data)
                area_dict["SBAND_data"] = closest_data
                area_parameters.append(area_dict)
            capacities= self.HAPS_Channel_model.generate_capacities(area_parameters,centroid_point, 8, 16)
            #print(capacities)
            #self.final_capacities = capacities    
            #print(np.mean(capacities)/ np.min(capacities)* np.std(capacities) , np.sum(capacities))
            #print(entropy(capacities))
            minimum_capacity = np.min(capacities)
            min_constraint = 15 
            penalty = 1000 if minimum_capacity < min_constraint else 0
            #capacities_values = area_values * capacities
            objective = np.sum(capacities) -  0.7 * np.std(capacities) *  np.mean(capacities)/ np.min(capacities) - penalty
            #objective = np.sum(capacities) - 0.7* np.std(capacities) *  np.mean(capacities)/ np.min(capacities)
            #print(objective, ChosenAreas)
            final_objective += -objective  
            
        return final_objective
    
                
    def calculate_capacities(self, haps_position, area_points, ChosenAreas)-> None:

        # Convert to Cartesian coordinates
        ## normalize the weights
    
   
        
        z_value = 0  # You can change this to any value or logic you want for the z-coordinate.
        area_points_with_z = [(x, y, z_value) for x, y in area_points]
        
        area_parameters = []
        choosen_areas_types  = self.area_type_array[ChosenAreas]
        choosen_indoor_users_percentage = self.indoor_users_array[ChosenAreas]
        choosen_traditional_building_percentage = self.traditional_building_percentage_array[ChosenAreas]
        #print(choosen_areas_types)
        #z_value = 0  # You can change this to any value or logic you want for the z-coordinate.
        
        #haps_position = ( centroid_x, centroid_y, 20000)
        for coordinate,area_type, indoor_percentage, traditional_building  in zip(area_points_with_z, choosen_areas_types, choosen_indoor_users_percentage, choosen_traditional_building_percentage):
            area_dict = dict()
            coordinate = (coordinate[0]* 10**6, coordinate[1] * 10**6, 0)
            area_dict["area_coordinates"] = coordinate
            area_dict["area_type"] = area_type
            area_dict["indoor_percentage"] = indoor_percentage
            area_dict["traditional_building_percentage"] = traditional_building
            area_dict["N_sim"] = 100
            
            receiver_point = (coordinate[0], coordinate[1], 0)  # Example receiver point with z = 0
            centroid_point = (haps_position[0]*10**6, haps_position[1]*10**6, 20000)  # Centroid with z = 20000
            #print(receiver_point, centroid_point)   
            # Compute the horizontal distance between the receiver and the centroid
            horizontal_distance = np.sqrt((centroid_point[0] - receiver_point[0])**2 + (centroid_point[1] - receiver_point[1])**2)

            # Compute the height difference
            height_difference = centroid_point[2] - receiver_point[2]

            # Compute the elevation angle in radians
            elevation_angle_rad = np.arctan2(height_difference, horizontal_distance)

            # Convert the elevation angle to degrees
            elevation_angle_deg = np.degrees(elevation_angle_rad)
            #print(elevation_angle_deg)
            
            area_dict["elevation_angle_deg"] = elevation_angle_deg
            closest_data = min(self.haps_sband_parameters.data, key=lambda d: abs(d.elevation - elevation_angle_deg))
            #print(closest_data)
            area_dict["SBAND_data"] = closest_data
            area_parameters.append(area_dict)
            
        capacities= self.HAPS_Channel_model.generate_capacities(area_parameters,centroid_point, self.n_areas, 8)
        
        return capacities
    
    
    def plot_pie(self,solution, *args):
            """
            Constraint for a point to lie within a pie slice facing point A.

            Parameters:
                solution (list or array): [x, y] coordinates of the solution.
                A (tuple): Coordinates of point A (x1, y1).
                B (tuple): Coordinates of point B (x2, y2) - center of the circle.
                radius (float): Radius of the circle.

            Returns:
                list: A list of constraints where each element should be <= 0.
            """
            radius = 8_000
            A = self.bary_center
            B = self.haps_position
            #print("A", A)
            #print("B", B)
            #print("Solution", solution)
            x, y = solution
            x1, y1 = A
            x2, y2 = B
            x1 = x1 * 10**6
            y1 = y1 * 10**6
            x2 = x2 * 10**6
            y2 = y2 * 10**6
            x = x * 10**6
            y = y * 10**6
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < radius/2 : 
                radius = distance + 1000
          
            # Distance constraint: within the circle
            dist_constraint = np.sqrt((x - x2)**2 + (y - y2)**2) - radius
            A = [x1, y1]
            B = [x2, y2]
            solution = [x, y]
            # Angle from B to A (center of the pie slice)
            theta_center = np.arctan2(y1 - y2, x1 - x2)
            theta_min = theta_center - np.pi / 6
            theta_max = theta_center + np.pi / 6

            # Angle from B to the solution
            theta = np.arctan2(y - y2, x - x2)

            ## Normalize theta to be within [theta_min, theta_max]
            #if theta < theta_min:
            #    theta += 2 * np.pi
            #elif theta > theta_max:
            #    theta -= 2 * np.pi

            # Angular constraints
            angular_constraint_min = theta_min - theta
            angular_constraint_max = theta - theta_max
            plt.figure(figsize=(8, 8))

            # Plot the circle
            theta = np.linspace(0, 2 * np.pi, 500)
            circle_x = B[0] + radius * np.cos(theta)
            circle_y = B[1] + radius * np.sin(theta)
            plt.plot(circle_x, circle_y, linestyle="--", color="gray", label="Circle")

            # Plot the pie slice
            theta_min = np.arctan2(A[1] - B[1], A[0] - B[0]) - np.pi / 6
            theta_max = np.arctan2(A[1] - B[1], A[0] - B[0]) + np.pi / 6
            pie_theta = np.linspace(theta_min, theta_max, 500)
            pie_x = [B[0]] + list(B[0] + radius * np.cos(pie_theta)) + [B[0]]
            pie_y = [B[1]] + list(B[1] + radius * np.sin(pie_theta)) + [B[1]]
            plt.fill(pie_x, pie_y, color="orange", alpha=0.5, label="Pie Slice Facing A")

            # Plot points A and B
            plt.scatter(*A, color="blue", label="Point A")
            plt.scatter(*B, color="red", label="Point B (Center of Circle)")

            # Plot test points

            #color = "green" if flags else "purple"
            #label = f"Test Point {i+1} ({'Valid' if flags[i] else 'Invalid'})"
            plt.scatter(x, y)

            # Configure plot
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title("Constraint Verification: Pie Slice Facing Point A")
            plt.legend()
            plt.grid()
            plt.axis("equal")
            plt.show()
            pie_x = [B[0]] + list(B[0] + radius * np.cos(pie_theta)) + [B[0]]
            pie_y = [B[1]] + list(B[1] + radius * np.sin(pie_theta)) + [B[1]]
            plt.fill(pie_x, pie_y, color="orange", alpha=0.5, label="Pie Slice Facing A")

            # Plot points A and B
            plt.scatter(*A, color="blue", label="Point A")
            plt.scatter(*B, color="red", label="Point B (Center of Circle)")

            # Plot test points
            
            #color = "green" if flags[i] else "purple"
            #label = f"Test Point {i+1} ({'Valid' if flags[i] else 'Invalid'})"
            plt.scatter(x, y)

            # Configure plot
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title("Constraint Verification: Pie Slice Facing Point A")
            plt.legend()
            plt.grid()
            plt.axis("equal")
            plt.show()

         
            