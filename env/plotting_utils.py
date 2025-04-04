import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import List 
import seaborn as sns










def plot_over_capacity(over_capacity_old : List[int] , over_capacity_new: List[int], n_areas : int , area_names : List[str])-> None: 
    
    
    over_capacity_old_array = np.array(over_capacity_old)
    area_with_over_capacity_old = np.argwhere(over_capacity_old_array > 0)
    over_capacity_new = np.array(over_capacity_new)[area_with_over_capacity_old].flatten()
    area_names = np.array(area_names)
    area_names = area_names[area_with_over_capacity_old].flatten()
    over_capacity_old = over_capacity_old_array[area_with_over_capacity_old].flatten()

    plt.figure(figsize=(13, 6))
    x_values = np.arange(len(area_with_over_capacity_old))
    bar_width = 0.35
    plt.bar(x_values, over_capacity_old, width=bar_width, capsize=5, label='Without HAPS',
                color='darkred', alpha=0.7)
    plt.bar(x_values + bar_width, over_capacity_new, width=bar_width, capsize=5, label='With HAPS',
            color='skyblue', alpha=0.7)
    plt.xlabel('Areas')
    plt.ylabel('Over Capacity')
    plt.title('Over Capacity Per Area')
    plt.xticks(x_values, area_names, rotation=45, fontsize="small")
    plt.legend()
    plt.grid(axis='y')
    plt.savefig('results/over_capacity_per_area.png')
    plt.show()
def cdf_plot(old_traffic, new_traffic ): 
    #def compute_cdf(data):
    #    sorted_data = np.sort(data)
    #    cdf = np.arange(1, len(data) + 1) / len(data)
    #    return sorted_data, cdf
#
# Co#mpute CDFs
    #sorted_a, cdf_a = compute_cdf(old_traffic)
    #sorted_b, cdf_b = compute_cdf(new_traffic)
#
    ## Plotting the CDFs
    #plt.figure(figsize=(10, 6))
    #plt.plot(sorted_a, cdf_a, label='Without HAPS')
    #plt.plot(sorted_b, cdf_b, label='WIth HAPS')
    #plt.title('Cumulative Distribution Functions (CDFs)')
    #plt.xlabel('Value')
    #plt.ylabel('CDF')
    #plt.legend()
    #plt.grid(True)
    #plt.show()
    #generate bins and bin edges

    #cumsum and normalize to get cdf rather than pdf
    plt.figure(figsize=(10, 6))
    sns.kdeplot(old_traffic, cumulative=True, bw_adjust=0.2, label='No HAPS')
    sns.kdeplot(new_traffic, cumulative=True, bw_adjust=0.2, label='Smooth CDF')
    plt.title('Smooth CDF Using Seaborn')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.show()
 
def plot_load_bar(old_load_report_mean : List[int],new_load_report_mean : List[int],ideal_report_mean: list[int],n_areas : int , area_names : List[str])-> None:
    
    plt.figure(figsize=(13, 6))
    
 
    n = len(area_names) 
    # Width of each bar
    bar_width = 0.4

    x_values = np.arange(0, n * 2, 2)
  
    plt.bar(x_values- bar_width , old_load_report_mean , width=bar_width, capsize=5, label='Load before off loading', color='darkred', alpha=0.7)
    plt.plot(x_values- bar_width, old_load_report_mean, color='darkred', alpha=0.7)
    plt.bar(x_values , new_load_report_mean,width=bar_width, capsize=5, label='Load After off loading', color='skyblue', alpha=0.7)
    # Bars without standard deviation
    plt.plot(x_values, new_load_report_mean, color='skyblue', alpha=0.7)
    #plt.bar(x_values + bar_width, ideal_report_mean, width=bar_width, label='Load without peaks (Upper Bound))', color='orange', alpha=0.7)

    # Adding labels and title
    plt.xlabel(' Area name')
    plt.ylabel('Average Load')
    plt.title('Average Load Per Area')
    plt.xticks(x_values, area_names, rotation = 45,fontsize = "small" )   # Set the position of the x ticks
    plt.ylim(0, 0.8)
    plt.legend()

    plt.grid(axis='y')
    plt.savefig('results/mean_load_per_area_2_ll_new.png')
    plt.show()  

def plot_load_boxplot(old_load_report : List[List], new_load_report : List[List], n_areas : int , area_names : List[str])-> None:
    plt.figure(figsize=(13, 6))

    # Create boxplots
    box_width = 0.35
    x_values = np.arange(len(area_names))
    positions_old = x_values - box_width / 2
    positions_new = x_values + box_width / 2
    for i in range(len(old_load_report)):
        print("Area Old", i, np.median(old_load_report[i]),np.mean(old_load_report[i]), np.min(old_load_report[i]))
        print("Area New", i, np.median(new_load_report[i]),np.mean(new_load_report[i]), np.min(new_load_report[i]))
    bp_old = plt.boxplot(old_load_report, positions=positions_old, showfliers=False, widths=box_width, patch_artist=True,
                        boxprops=dict(facecolor='darkred', color='darkred', alpha=0.7),
                        capprops=dict(color='darkred'),
                        whiskerprops=dict(color='darkred'),
                        flierprops=dict(marker='o', color='darkred', alpha=0.7),
                        medianprops=dict(color='yellow'))

    bp_new = plt.boxplot(new_load_report, positions=positions_new, showfliers=False,widths=box_width, patch_artist=True,
                        boxprops=dict(facecolor='skyblue', color='skyblue', alpha=0.7),
                        capprops=dict(color='skyblue'),
                        whiskerprops=dict(color='skyblue'),
                        flierprops=dict(marker='o', color='skyblue', alpha=0.7),
                        medianprops=dict(color='blue'))

    # Adding labels and title
    plt.xlabel('Areas')
    plt.ylabel('Load')
    plt.title('Load per area')
    plt.xticks(x_values, area_names, rotation=45, fontsize="small")  # Set the position of the x ticks
    plt.legend([bp_old["boxes"][0], bp_new["boxes"][0]], ['Without HAPS', 'With HAPS'])
    plt.grid(axis='y')
    plt.savefig('results/mean_load_per_area_with_std_ll.png')
    plt.show()
def plot_traffic_distribution(): 


    # Generate some sample data if you haven't already defined traffic_sum
    test_traffic = trace_matrix[:,train_len:]
    traffic_sum = np.sum(test_traffic, axis=1)
    tot_traffic_pd = pd.DataFrame({"Traffic Load":traffic_sum})

    data_sorted = tot_traffic_pd.sort_values(by='Traffic Load', ascending=False)

    data_sorted = data.sort_values(by='Traffic Load', ascending=False)

    # Normalize the traffic load to a probability
    data_sorted['Normalized Load'] = data_sorted['Traffic Load'] / data_sorted['Traffic Load'].sum()

    # Calculate the cumulative percentage of normalized load
    data_sorted['Cumulative %'] = data_sorted['Normalized Load'].cumsum()

    # Create a new column for plotting that converts the index to a percentage of the total base stations
    data_sorted['Percent of Base Stations'] = 100 * data_sorted.reset_index().index / len(data_sorted)

    # Calculate the PDF by differentiating the CDF
    # Use np.diff to find the difference between consecutive cumulative values, append a 0 to align lengths
    data_sorted['PDF'] = np.append(np.diff(data_sorted['Cumulative %']), 0)  # Append 0 at the end to match lengths
    smoothed_pdf = gaussian_filter1d(data_sorted['PDF'], sigma=2)  # Smoothing the PDF

    # Plotting with different colors for different ranges
    plt.figure(figsize=(10, 5))
    colors = ['blue', 'green', 'purple']
    break_points = [10, 30, 100]

    # Plot segments
    for i, color in enumerate(colors):
        if i == 0:
            mask = (data_sorted['Percent of Base Stations'] <= break_points[i])
            label = "High Load"
        elif i < len(colors) - 1:
            mask = (data_sorted['Percent of Base Stations'] >= break_points[i-1]) & (data_sorted['Percent of Base Stations'] <= break_points[i])
            label = "Medium Load"
        else:
            mask = (data_sorted['Percent of Base Stations'] >= break_points[i-1])
            label = "Low Load"

        plt.fill_between(data_sorted.loc[mask, 'Percent of Base Stations'], smoothed_pdf[mask], color=color, alpha=0.5, label=label)
    # Annotations
    plt.annotate('', xy=(0, 0.03), xytext=(30, 0.03),
                arrowprops=dict(arrowstyle='<->', lw=2),
                )
    plt.annotate('', xy=(30, 0.03), xytext=(100, 0.03),
                arrowprops=dict(arrowstyle='<->', lw=2),
                )

    plt.text(15, 0.03, '~60% of total traffic', verticalalignment='bottom', horizontalalignment='center', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.text(60,0.03,'~40% of total traffic', verticalalignment='bottom', horizontalalignment='center', fontdict={'fontsize': 12, 'fontweight': 'bold'})

    #ax.annotate('', xy=(dates[0], first_half_mean), xytext=(dates[half_point], first_half_mean),
    #            arrowprops=dict(arrowstyle='<->', lw=2))
    #ax.annotate('', xy=(dates[half_point], second_half_mean), xytext=(dates[-1], second_half_mean),
    #            arrowprops=dict(arrowstyle='<->', lw=2))

    #ax.text(dates[half_point // 2], first_halfean, 'Initialization Phase', verticalalignment='bottom', horizontalalignment='center', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    #ax.text(dates[half_point + (len(dates) - half_point) // 2], second_half_mean, 'Interaction Phase', verticalalignment='bottom', horizontalalignment='center', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    # Additional Plot Formatting
    plt.title('Probability Density Function (PDF) for Load by Site Distribution')
    plt.xlabel('Percentage of Base Stations')
    plt.ylabel('Probability Density (%)')
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0, max(smoothed_pdf) + 0.01)  # Adjust y-axis to fit the smoothed data
    plt.xticks([0, 10, 30, 100], ['0%', '10%', '30%', '100%'])
    plt.legend()
    plt.tight_layout()
    plt.show()