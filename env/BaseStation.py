import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import datetime
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import matplotlib.pyplot as plt
from typing import Optional
import os 
import matplotlib.dates as mdates
import logging


class BaseStation(object):
    """ BaseStation class for base station object
    
    Args:
        traffic (np.ndarray): traffic data of base station
        cell_number (str): cell number of base station    
    """

    def __init__(self, traffic : np.ndarray, cell_number : str, capacity : float, load_level : str, algo : str)-> None:
        self.traffic = traffic
        self.cell_number = cell_number
        initial_date = datetime.datetime(2022,4,3)
        self.time = pd.date_range(start=initial_date, periods=len(traffic), freq="h")
        self.load_level = load_level
        self.capacity = capacity
        self.algo = algo 
        
    def __len__(self) -> int : 
        if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3"or self.algo == "greedy":
            return len(self.traffic)
        else : 
            return self._test_len
    
    def TrafficMetrics(self): 
        DailyLoad = list()
        NewDailyLoad = list()
        new_peak_traffic = 0
        old_peak_traffic = 0
        ideal_peak_traffic = 0
        tot_capacity = 0
        
        IdealLoad = list()
        for Ttest in range(self._test_len): 
            if self.algo == "epsilon-greedy" or self.algo == "ucb" or "exp3" or self.algo == "greedy": 
                T = Ttest+ self._train_len
            else : 
                T = Ttest + self._train_len
            T = Ttest + self._train_len 
            t = self.time[T]
            if t.hour <8 : 
                continue 
            #if self.traffic[T] > 0.6*self.capacity:
            DailyLoad.append(self.traffic[T]/self.capacity)
            IdealLoad.append(self.ideal_new_traffic[T]/self.capacity)
            NewDailyLoad.append(self.new_traffic[T]/self.capacity)   
            old_peak_traffic +=self.traffic[T]
            new_peak_traffic += self.new_traffic[T]
            ideal_peak_traffic += self.ideal_new_traffic[T]
            tot_capacity += self.capacity        

        return  DailyLoad, \
                NewDailyLoad, \
                IdealLoad, \
                old_peak_traffic,\
                new_peak_traffic,\
                ideal_peak_traffic,\
                tot_capacity

    def ideal_off_loading_decision(self, t_int : int, choosen : bool)-> float :
        #print("Ideal off loading decision")
        if choosen:
            if self.traffic[t_int if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy" else t_int + self._train_len ] > 0.6*self.capacity:
                self.ideal_new_traffic[t_int] = self.get_upper_bound(t_int, use_normalized_traffic = False)
                #self.peak_occurence_ideal[t_int] += (self.traffic[t_int + self._train_len]-self.ideal_new_traffic[t_int])/self.capacity
            
            else : 
                self.ideal_new_traffic[t_int] = self.traffic[t_int if self.algo == "epsilon-greedy" or  self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy" else t_int + self._train_len ]
        else : 
            
            self.ideal_new_traffic[t_int] = self.traffic[t_int if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy" else t_int + self._train_len ]
            if self.traffic[t_int if self.algo == "epsilon-greedy" or self.algo == "ucb"  or self.algo == "exp3"  or self.algo == "greedy" else t_int + self._train_len ] > 0.6*self.capacity:
                self.peak_occurence_ideal[t_int] += (self.traffic[t_int if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy" else t_int + self._train_len ]-self.get_upper_bound(t_int, use_normalized_traffic=False))/self.capacity
        

        
    def get_traffic(self, t_int : int) -> float :
        if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy": 
            return  self.traffic[t_int]
        else : 
            return self.traffic[t_int + self._train_len]
    
    def get_new_traffic(self, t_int : int)-> float :
        return self.new_traffic[t_int]

    def off_load_in_t(self, t_int : int, chosen : bool = False)-> float :
        self.upper_bound[t_int] = self.get_upper_bound(t_int, use_normalized_traffic = False)
        if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy": 
            t = t_int
        else : 
            t = t_int + self._train_len
        real_traffic = self.traffic[t]
        self.flag = False
        self.flag_ol = False
        #logging.info(f"off loading for bs {self.cell_number} at time {self.time[t]}")
        #logging.info(f"traffic {real_traffic:e} upper bound {self.upper_bound[t_int]:e}")
        #logging.info(f"{self.hourly_ps_memory}")
        if chosen : 
            logging.info("Chosen")
            if real_traffic > 0.6*self.capacity:
                #logging.info(f"off loading for bs {self.cell_number} at time {self.time[t]}")
                #logging.info(f"traffic {real_traffic:e} upper bound {self.upper_bound[t_int]:e}")
                self.new_traffic[t_int] = 0.6*self.capacity
                if real_traffic > 0.6*self.capacity:
                    self.peaks[t_int] += 1
                    self.peak_occurence[t_int] +=(real_traffic-self.upper_bound[t_int])/self.capacity
                if real_traffic > 0.8*self.capacity:    
                    self.flag = True

            else : 
                self.new_traffic[t_int] = real_traffic
            
        else : 
            self.new_traffic[t_int] = real_traffic
            
            if real_traffic > 0.6*self.capacity:
            
                self.peak_occurence[t_int] +=(real_traffic-self.upper_bound[t_int])/self.capacity
                self.peak_occurence_ol[t_int] += (self.new_traffic[t_int]-self.upper_bound[t_int])/self.capacity   
                self.peaks[t_int] += 1
                self.peaks_ol[t_int] += 1
            if real_traffic > 0.8*self.capacity:
                self.flag = True
                self.flag_ol = True
        diff = (real_traffic - self.upper_bound[t_int])
        t_temporal = t_int - self._train_len
        self.old_load[t_temporal] = real_traffic  /self.capacity
        self.new_load[t_temporal] = self.new_traffic[t]/self.capacity
        return  diff if diff > 0 else 0 , diff/self.capacity if diff > 0 else 0, real_traffic if diff > 0 else 0

    
    def save_load(self, path : str)-> None:
        avg_traffic_old = np.mean(self.traffic[self._train_len:])
        avg_traffic_new = np.mean(self.new_traffic)
        max_traffic_old = np.max(self.traffic[self._train_len:])
        max_traffic_new = np.max(self.new_traffic)
        load_report_bs = np.zeros(4)
        traffic_report = np.zeros(4)
      
        load_report_bs[:] = (avg_traffic_old/self.capacity, avg_traffic_new/self.capacity,\
                              max_traffic_old/self.capacity, max_traffic_new/self.capacity)

        traffic_report[:] = (avg_traffic_old, avg_traffic_new, max_traffic_old, max_traffic_new)
        np.save(os.path.join(path,f"{self.cell_number}_load_report.npy"), load_report_bs)
        np.save(os.path.join(path,f"{self.cell_number}_traffic_report.npy"), traffic_report)
    

    def MakePlots(self,path : str)-> None :
        # Plot the different traffic series

        plt.figure(figsize=(20, 10))
        self._train_len = 0
        old_traffic = self.traffic[self._train_len:]
        print(len(self.time[self._train_len:]), len(self.upper_bound), len(old_traffic), len(self.new_traffic), len(self.ideal_new_traffic))
        #plt.plot(self.time[self._train_len:], self.upper_bound, label="Upper Bound", color="black")
        plt.hlines(0.6*self.capacity, self.time[self._train_len], self.time[-1], color = "black", linestyle = "--", label = "Upper Bound")
        plt.plot(self.time[self._train_len:], old_traffic, label="Old Traffic")
        plt.plot(self.time[self._train_len:], self.new_traffic, label="Traffic", color="red")
        plt.plot(self.time[self._train_len:], self.ideal_new_traffic, label="Ideal Traffic", color="green")

        # Format x-axis to show hours
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=10))

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=90, ha='right', fontsize=8)
        plt.axhline(self.capacity, color = "black", linestyle = "--", label = "Capacity")
        plt.grid()
        plt.xlabel("dates")

        plt.ylabel("Traffic in Bits/hour")
        plt.legend()
        plt.title(f"Base Station {self.cell_number}, Load Level {self.load_level}")
        plt.savefig(os.path.join(path,f"{self.cell_number}.png"))
        plt.close()
        
    
    def update_ps_buffer(self, t_int : int )-> None : 
        if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy": 
            t = t_int
        else : 
            t = t_int + self._train_len
        
        upper_bound = self.get_upper_bound(t_int, log = False)
        time = self.time[t]

        if time.hour == 8 :
            self.hourly_ps_memory.append([upper_bound, upper_bound])
            self.hourly_ps_memory.append([upper_bound, self.traffic_norm[t]])
        else : 
            self.hourly_ps_memory.append([upper_bound, self.traffic_norm[t]])


    def get_hourly_ps(self)-> Optional[float] :
        alpha = 1
        new_traffic = list(map(lambda x : x[0] if x[0] < x[1] else x[1], self.hourly_ps_memory))
        old_traffic = list(map(lambda x : x[1], self.hourly_ps_memory))
        #if old_traffic[0] > 0.6*self.capacity and old_traffic[1] > 0.6*self.capacity and old_traffic[0] > old_traffic[1]:
        #    old_traffic[0] = old_traffic[1]
        new_traffic_interpolation = UnivariateSpline(np.arange(2), new_traffic, s=0, k=1)
        old_traffic_interpolation = UnivariateSpline(np.arange(2), old_traffic, s=0, k=1)
        x_dense = np.linspace(0,1,200)
        new_traffic_dense = new_traffic_interpolation(x_dense)
        old_traffic_dense = old_traffic_interpolation(x_dense)
        hourly_ops =   old_traffic_dense - new_traffic_dense
        hourly_ops[hourly_ops < 0] = 0
        #if old_traffic[0] > 0.6*self.capacity and old_traffic[1] < 0.6*self.capacity:
        #    return  0 
        #integral = np.trapz(hourly_ops) / (self.capacity*2)
           # Define the weighting function (higher weights when old traffic > 0.6 * capacity)
        weight = 1 + alpha * np.maximum(0, (old_traffic_dense - 0.6 * self.capacity) / self.capacity)
    
        # Weighted integration
        integral = np.trapz(hourly_ops * weight) / (self.capacity * 2)
    
        # Special case: if old traffic drops below 0.6 * capacity
        if old_traffic[0] > 0.6 * self.capacity and old_traffic[1] < 0.6 * self.capacity:
            return 0 
        return integral       
    
    
    def normalize(self) -> None: 
        self.scaler = MinMaxScaler()
        new_traffic = np.append(self.traffic, self.capacity).flatten()
        self.scaler.fit(new_traffic.reshape(-1,1))
        self.capacity_norm = self.scaler.transform(new_traffic.reshape(-1,1))[-1][0]
        self.traffic_norm = self.scaler.transform(self.traffic.reshape(-1,1)).flatten()
        self.traffic_norm  = self.traffic 
        
  
    def get_upper_bound(self, t_int : int = 0, w : int = 24, scale : float = 1, use_normalized_traffic : bool = True, log : bool = False) -> float : 
  
        if use_normalized_traffic : 
            
            return 0.6*self.capacity     
        else :
            
            return 0.6*self.capacity
 
    def train_test_split(self, split : float = 0.5) -> None : 

        if hasattr(self,"traffic_norm"): 
            #traffic = self.traffic_norm.flatten()      
            traffic = self.traffic

        else : 
             raise RuntimeError("You must first call normalize() method")
        n_observation = len(traffic)   
        self._train_len = int(n_observation*split) - int(n_observation*split)%24
        self._test_len = n_observation - self._train_len
        self._train_traffic = traffic[:self._train_len]
        
        #if self.cell_number > 41 and self.cell_number < 49:
        #    plt.figure(figsize=(20,10))
        #    
        #    
        #    plt.plot( self.traffic_norm[self._train_len:], label = "Traffic",color = "red")
        #    
        #    
        #    plt.axhline(self.capacity_norm, color = "black", linestyle = "--", label = "Capacity")
        #    plt.axhline(0.6*self.capacity_norm, color = "black", linestyle = "--", label = "Upper Bound")
        #    plt.xlabel("dates")
        #    plt.ylabel("Traffic in Bits/hour")
        #    plt.legend()
        #    plt.title(f"Base Station {self.cell_number}, Load Level {self.load_level}")
        #    plt.show()
            
    
    def get_stats(self, scale = 1, w = 24, day_split = 8) -> Optional[pd.DataFrame]: 
        
        def get_part_of_day(time):
            hour = time.hour
            if 8 <= hour < 16:
                return 'Day'
            
            elif 16 <= hour <= 23:
                return 'Night'
        

        self.normalize()
        self.train_test_split()
        
        traffic = pd.Series(self._train_traffic)
        
        
     
        anomalies = pd.Series(index=traffic.index)
        new_traffic = traffic.copy()
        
        
        min_limit = (0.6 * self.capacity).astype(np.float32)
        new_traffic[traffic > min_limit] = min_limit
        anomalies[traffic>min_limit] = 1
        
        y = traffic.values
        y_1 = new_traffic.values
        j = 0 
        stats_diff = pd.DataFrame()
        for i in range(0, len(y_1),24):

                time_window = self.time[i+8:i+24]
                j+=1
                y_1_daily = y_1[i+8:i+24]
                y_daily = y[i+8:i+24]
                #diff_window = deriv_diff[i+8:i+24]
                anomalies_window = anomalies[i+8:i+24]
                std_traffic = np.std(y[i+8:i+24])
             
                for j in range(0, len(y_daily),day_split):
                        y_1_part = y_1_daily[j:j+day_split]
                        y_part = y_daily[j:j+day_split]
                        inter_y_1 = UnivariateSpline(np.arange(len(y_1_part)),y_1_part,s=0,k=1)
                        inter_y = UnivariateSpline(np.arange(len(y_part)),y_part,s=0,k=1)
                        x_dense = np.linspace(0,day_split-1,1000)
                        y_dense = inter_y(x_dense)
                        y_1_dense = inter_y_1(x_dense)
                        sub_diff_window = y_dense - y_1_dense

                        sub_time_window = time_window[j:j+day_split]
                        
                        integral = np.trapz(sub_diff_window) / self.capacity
                        
                        start_date = sub_time_window[0]
                        end_date = sub_time_window[-1]
                        sub_anomalies_window = anomalies_window[j:j+day_split]
                        sum_anomalies = sub_anomalies_window.sum()
                       
                        stats_diff = pd.concat([stats_diff, pd.DataFrame({"start_date":start_date, 
                                                                          "end_date":end_date,
                                                                          "integral": integral
                                                                          , "sum_anomalies":sum_anomalies, 
                                                                          "std_traffic": std_traffic}, index=[0])])
                                                                          
                        
        if stats_diff.empty : 
            return None              
        stats_diff['start_date'] = pd.to_datetime(stats_diff['start_date'])
        
        # Extract day of the week from 'start_date'
        stats_diff['day'] = stats_diff['start_date'].dt.day_name()
        # Apply the function to 'start_date' to get part of the day
        stats_diff['part_of_day'] = stats_diff['start_date'].apply(get_part_of_day)
        
        
        stats_diff.groupby(['day','part_of_day']).agg({
            'integral':'sum','sum_anomalies':'sum'}).reset_index()
        return stats_diff
    def get_energy_stats(self): 
        self.normalize()
        self.train_test_split()
        
        traffic = pd.Series(self._train_traffic)
        
        
        initial_date = datetime.datetime(2022,4,1)
        time = pd.date_range(start=initial_date, periods=len(traffic), freq="h")
        new_traffic = traffic.copy()
        
        
        energy_threshold = (0.15 * self.capacity_norm).astype(np.float32)
        new_traffic[traffic < energy_threshold] = energy_threshold
        
        
        y = traffic.values
        y_1 = new_traffic.values
          
        energy_stats = pd.DataFrame()
        for i in range(0, len(y_1),24):
                time_window = time[i+1:i+8]
                
                y_1_daily = y_1[i+1:i+8]
                y_daily = y[i+1:i+8]
                #diff_window = deriv_diff[i+8:i+24]
                inter_y = UnivariateSpline(np.arange(len(y_daily)),y_daily,s=0,k=1)
                inter_y_1 = UnivariateSpline(np.arange(len(y_1_daily)),y_1_daily,s=0,k=1)
                x_dense = np.linspace(0,7-1,1000)
                y_dense = inter_y(x_dense)
                y_1_dense = inter_y_1(x_dense)
                sub_diff_window = y_1_dense - y_dense
                sub_diff_window[sub_diff_window < 0] = 0
                integral = np.trapz(sub_diff_window)
                energy_stats = pd.concat([energy_stats, pd.DataFrame({"start_date":time_window[0], 
                                                                  "end_date":time_window[-1],
                                                                  "integral": integral}, index=[0])])
        
                                                                    
                        
        if energy_stats.empty : 
            return None              
        energy_stats['start_date'] = pd.to_datetime(energy_stats['start_date'])
        
        # Extract day of the week from 'start_date'
        energy_stats['day'] = energy_stats['start_date'].dt.day_name()
        # Apply the function to 'start_date' to get part of the day
        
        
        
        energy_stats.groupby('day').agg({
            'integral':'sum'}).reset_index()
        
        return energy_stats
    
    
        
    def reset(self)-> None: 
        if self.algo == "epsilon-greedy" or self.algo == "ucb" or self.algo == "exp3" or self.algo == "greedy": 
            length = len(self.traffic)
        else : 
            length = self._test_len
        self.hourly_ps_memory = deque(maxlen=2) 
        self.ideal_new_traffic = np.zeros(length)
        self.new_traffic = np.zeros(length)
        
        self.upper_bound = np.zeros(length)
        self.test_time = self.time[:-length]
        self.peak_occurence = np.zeros(length)
        self.peak_occurence_ol = np.zeros(length)
        self.peak_occurence_ideal = np.zeros(length)
        self.peaks = np.zeros(length)
        self.peaks_ol = np.zeros(length)
        self.new_load = np.zeros(self._test_len+1)
        self.old_load = np.zeros(self._test_len+1)
        
        
           
