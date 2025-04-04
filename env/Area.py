from env.BaseStation import BaseStation
import pandas as pd
import numpy as np
import logging
import os 
from typing import Tuple
from itertools import chain
from collections import deque
class Area(object): 
    def __init__(self, area_name : str, area_type : str , indoor_users : float, traditional_building_percentage: float)-> None: 
        self.area_name = area_name
        self.base_stations = [] 
        self.area_type = area_type
        self.hps_buffer = deque(maxlen = 24)
        self.hps_window = dict()
        self.indoor_users = indoor_users
        self.traditional_building_percentage = traditional_building_percentage
        
        
    def add_base_station(self, base_station : BaseStation)-> None: 
        self.base_stations.append(base_station)
    
    def __len__(self)-> int:
        return len(self.base_stations)
    
    def __contain__(self, base_station : BaseStation)-> bool: 
        return base_station in self.base_stations
    
    def __getitem__(self, index : int)-> BaseStation:
        return self.base_stations[index]
    
    def __iter__(self)-> BaseStation:
        return iter(self.base_stations)
    

    
    
    def update_ps_buffer_area(self, time : int)-> None:
                
         for bs in self: 
            bs.update_ps_buffer(time)
         #if self.area_name == "Linate":
         #   
         #   for bs in self: 
         #       logging.info(f"Lin update ps buffer {bs.hourly_ps_memory}")
     
    def area_ideal_decision(self, time : int, choosen : bool)-> None:
        for bs in self: 
            bs.ideal_off_loading_decision(time, choosen)
    
    def get_hourly_ps(self)-> float :
        #if self.area_name == "Linate": 
        #    for bs in self: 
        #        logging.info(f"Lin HPS {bs.get_hourly_ps()}")
        hourly_ops = np.sum([ bs.get_hourly_ps()  for bs in self])
        self.hps_buffer.appendleft(hourly_ops)
        return hourly_ops/ len(self)

    def critical_area_load(self):    
        if np.sum([ bs.flag for bs in self]) > 0:
            print("Critical Area Load", self.area_name, self.critical_load) 
            self.critical_load += 1
        if np.sum([ bs.flag_ol for bs in self]) > 0:
            print("Critical Area Load OL", self.area_name, self.critical_load) 
            self.critical_load_ol += 1

    def post_processing(self,t_int : int, chosen : bool)-> Tuple[int, float]:
        #if self.area_name == "Linate": 
        #    for bs in self: 
        #        logging.info(f"Lin off load {bs.off_load_in_t(t_int, chosen)}")
        result = np.array([ bs.off_load_in_t(t_int, chosen) for bs in self])
        traffic_diff = np.array(list(map(lambda x : x[0], result)))
        load_reduction = np.array(list(map(lambda x : x[1], result)))
        bs_with_peaks = len(traffic_diff[traffic_diff > 0])
        off_bound_traffic = np.sum(traffic_diff)
        self.critical_area_load()
        total_load_reduction = np.sum(load_reduction)
        self.excess_bs_list.append(bs_with_peaks)
        self.excess_traffic_list.append(off_bound_traffic)
        
        self.area_tot_traffic.append(np.sum(np.array(list(map(lambda x : x[2], result))))) 
        if chosen: 
            self.excess_bs_list_ol.append(0)
            self.excess_traffic_list_ol.append(0)
        else :
            self.excess_bs_list_ol.append(bs_with_peaks)
            self.excess_traffic_list_ol.append(off_bound_traffic)
        return bs_with_peaks, off_bound_traffic, total_load_reduction
    
    def update_area_traffic(self, time_int)-> Tuple[int, int]:
        self.area_traffic.append(np.sum([ bs.get_traffic(time_int) for bs in self]))
        self.new_area_traffic.append(np.sum([ bs.get_new_traffic(time_int) for bs in self]))
    
    
    def get_area_energy_stats(self)-> pd.DataFrame:
        section_stats_energy = []
        for bs in self:
            energy_bs = bs.get_energy_stats()
            if energy_bs is None:
                continue
            section_stats_energy.append(energy_bs)
        df_section_energy = pd.concat(section_stats_energy, axis = 0)
        df_agg_energy  = df_section_energy.groupby('day').agg({'integral': 'sum'}).reset_index()
        df_agg_energy.set_index("day", inplace=True)
        
        return df_agg_energy

    def get_area_stats(self)-> pd.DataFrame: 
        section_stats = []
        for bs in self: 
            bs_stats = bs.get_stats()
            if bs_stats is None:
                continue
            section_stats.append(bs_stats)
        df_section = pd.concat(section_stats, axis = 0)
        df_agg  = df_section.groupby(['day', 'part_of_day']).agg({'integral': 'sum', 'sum_anomalies': 'sum'}).reset_index()
        df_agg["day_part_of_day"] = df_agg["day"] + "-" + df_agg["part_of_day"]
        df_agg.set_index("day_part_of_day", inplace=True)
        ## get index as list 
        states_list = df_agg.index.tolist()
        for state in states_list : 
            self.hps_window[state] = deque(maxlen = 16)
        df_agg.drop(["day","part_of_day"], axis=1, inplace=True)
        
        return df_agg
    
    def plot(self)-> None: 
        area_plots_path = os.path.join("results","plots",self.area_name)
        os.makedirs(area_plots_path, exist_ok = True)
        for bs in self: 
            bs.MakePlots(area_plots_path)
        
    def FinalReport(self)-> None:
        LoadReport = [   bs.TrafficMetrics()  for bs in self]
        OldLoadReport = list(map( lambda x : x[0], LoadReport))
        NewLoadReport = list(map( lambda x : x[1], LoadReport))
        IdealLoadReport = list(map( lambda x : x[2], LoadReport))
        
        traffic_sum = sum(list(map( lambda x : x[3], LoadReport)))
        new_traffic_sum = sum(list(map( lambda x : x[4], LoadReport)))
        ideal_traffic_sum = sum(list(map( lambda x : x[5], LoadReport)))
        traffic_per_area = list(map( lambda x : x[3], LoadReport))
        new_traffic_per_area = list(map( lambda x : x[4], LoadReport))
        ideal_traffic_per_area = list(map( lambda x : x[5], LoadReport))
        capacity = sum(list(map( lambda x : x[6], LoadReport)))
        OldLoadReport = list(chain(*OldLoadReport))
        NewLoadReport = list(chain(*NewLoadReport)) 
        IdealLoadReport = list(chain(*IdealLoadReport))
        print(ideal_traffic_sum , new_traffic_sum, traffic_sum)
        return  OldLoadReport, \
                NewLoadReport,\
                IdealLoadReport,\
                traffic_per_area,\
                new_traffic_per_area,\
                ideal_traffic_per_area,\
                traffic_sum,\
                new_traffic_sum,\
                ideal_traffic_sum,\
                capacity
            

    def save_bs_load(self, path : str)-> None:
        for bs in self: 
            bs.save_load(path)
            
    def get_area_peak_occurence_report(self, test_len: int): 
        peak_occurence = np.zeros((len(self), test_len))
        peak_occurence_ol = np.zeros((len(self), test_len))
        peak_occurence_ideal = np.zeros((len(self), test_len))
        for index,bs in enumerate(self): 
            peak_occurence[index, :] = bs.peak_occurence
            peak_occurence_ol[index, :] = bs.peak_occurence_ol
            peak_occurence_ideal[index, :] = bs.peak_occurence_ideal

        return np.sum(peak_occurence, axis = 0), np.sum(peak_occurence_ol, axis = 0), np.sum(peak_occurence_ideal, axis = 0)
    
    def get_area_peak_report(self, test_len: int): 
        peak_occurence = np.zeros((len(self), test_len))
        peak_occurence_ol = np.zeros((len(self), test_len))
        for index,bs in enumerate(self): 
            peak_occurence[index, :] = bs.peaks
            peak_occurence_ol[index, :] = bs.peaks_ol
        return np.sum(peak_occurence, axis = 0), np.sum(peak_occurence_ol, axis = 0)

    def create_beam_data_frame(self, time)-> pd.DataFrame:
        self.data_rate = pd.Series( index = time)
        self.data_rate.fillna(0.0, inplace = True)
        
    
    
    def reset(self)-> None:
        self.area_traffic = []
        self.new_area_traffic = []   
        self.critical_load = 0   
        self.critical_load_ol = 0
        self.excess_bs_list = []
        self.excess_traffic_list = []
        self.excess_bs_list_ol = []
        self.excess_traffic_list_ol = []
        self.area_tot_traffic = []
        for bs in self: 
            bs.reset()
        

            