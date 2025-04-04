import numpy as np
import pandas as pd
from shapely.geometry import Point

import logging
from scipy.optimize import minimize



def distance(x, y, x_i, y_i):
    """Compute the Euclidean distance between two points."""
    return np.sqrt((x - x_i) ** 2 + (y - y_i) ** 2)

def objective_function(coords, cities, weights):
    """Objective function to minimize: weighted sum of distances."""
    x, y = coords
    return sum(weights[i] * distance(x, y, x_i, y_i) for i, (x_i, y_i) in enumerate(cities))

def calculate_los_angle(x, y, haps_x, haps_y):
    """Calculate the line-of-sight angle between two points."""
    delta_x = np.abs(x - haps_x)
    delta_y = np.abs(y - haps_y)
    delta = np.sqrt(delta_x ** 2 + delta_y ** 2)
    
    angle_rad = np.arctan2(delta, 20000)  # Angle in radians
    angle_deg = np.degrees(angle_rad)  # Convert to degrees
    
    return np.abs(angle_deg)  # Return the absolute value of the angle

def angle_constraint(coords, city, max_angle):
    """Constraint function to ensure the angle does not exceed the maximum."""
    x, y = coords
    x_i, y_i = city
    angle = calculate_los_angle(x, y, x_i, y_i)
    
    return max_angle - angle  # Constraint satisfied if this >= 0

def objective_with_penalty(coords, cities, weights, constraints, penalty_factor=1000):
    """
    Modified objective function that includes penalties for constraint violations.
    
    :param coords: Coordinates being optimized (x, y).
    :param cities: List of (x, y) tuples representing city coordinates.
    :param weights: List of weights for each city.
    :param constraints: List of constraint functions to check.
    :param penalty_factor: Factor that controls the penalty strength.
    :return: Objective value with penalty for constraint violations.
    """
    
    # Compute the original objective function (weighted sum of distances)
    obj_value = objective_function(coords, cities, weights)
    
    # Initialize penalty
    penalty = 0
    
    # Loop over each constraint and add penalty if violated
    for constraint in constraints:
        constraint_value = constraint['fun'](coords, *constraint['args'])
        if constraint_value < 0:  # If constraint is violated
            penalty += penalty_factor * (abs(constraint_value) ** 2)  # Quadratic penalty
    
    # Return the objective value plus penalties
    return obj_value + penalty

def weiszfeld_constrained(cities, weights, max_angles, max_iterations=10000, tolerance=1e-5, penalty_factor=100_000):
    """
    Compute the geometric median with angular constraints using constrained optimization with penalties.
    
    :param cities: List of (x, y) tuples representing city coordinates.
    :param weights: List of weights for each city.
    :param max_angle: Maximum allowable angle between the current point and each city.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :param penalty_factor: Penalty strength for violating constraints.
    :return: Estimated geometric median (x, y).
    """
    
    # Initial guess: centroid of the cities
    initial_guess = np.mean(cities, axis=0)
    #weights = np.log1p(weights)
    weights = np.ones(len(cities))
    # Define the angular constraints for each city
    constraints = [
        {
            'type': 'ineq',
            'fun': angle_constraint,
            'args': (city, max_angle)
        }
        for city, max_angle in zip(cities, max_angles)
    ]
    
    # Perform the constrained optimization with penalty-based objective
    result = minimize(
        objective_with_penalty,  # Using the penalty-based objective function
        initial_guess,
        args=(cities, weights, constraints, penalty_factor),
        tol=tolerance,
        options={'maxiter': max_iterations}, 
        method = "SLSQP"
          # You can try 'trust-constr' or 'COBYLA' too
    )
    
    # Check if the optimization was successful
    if not result.success:
        logging.info("Optimization did not converge successfully.")
        logging.info("Message:", result.message)
    logging.info(result.fun)
    # Return the optimized coordinates
    return result.x


''''
def distance(x, y, x_i, y_i):
    """Compute the Euclidean distance between two points."""
    return np.sqrt((x - x_i) ** 2 + (y - y_i) ** 2)

def objective_function(coords, cities, weights):
    """Objective function to minimize: weighted sum of distances."""
    x, y = coords
    return sum(weights[i] * distance(x, y, x_i, y_i) for i, (x_i, y_i) in enumerate(cities))

def calculate_los_angle(x , y, haps_x , haps_y):
        delta_x = np.abs(x - haps_x)
        delta_y = np.abs(y - haps_y)
        delta = np.sqrt(delta_x**2 + delta_y**2)
        
        angle_rad = np.arctan2(delta, 20000)  # Angle in radians
        angle_deg = np.degrees(angle_rad)  # Convert to degrees
        
        # Ensure angle is within 0 to 90 degrees
        angle_deg = np.abs(angle_deg)
            
        return angle_deg

def angle_constraint(coords, city, max_angle):
    """Constraint function to ensure the angle does not exceed the maximum."""
    x, y = coords
    x_i, y_i = city
    angle = calculate_los_angle(x, y, x_i, y_i)
    
    return max_angle - angle  # Constraint: max_angle >= angle

def weiszfeld_constrained(cities, weights, max_angles, max_iterations=1_000_000, tolerance=1e-15):
    """
    Compute the geometric median with angular constraints using constrained optimization.
    
    :param cities: List of (x, y) tuples representing city coordinates.
    :param weights: List of weights for each city.
    :param max_angle: Maximum allowable angle between the current point and each city.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :return: Estimated geometric median (x, y).
    """
    flag = False
    weights = np.log1p(weights)
    
    cities_index = np.arange(len(cities))
    
    while True :
        # Initial guess: centroid of the cities
        if flag:
            city_delete = np.argmin(lowest_constraints)
            cities_index = np.delete(cities_index, city_delete)
            del cities[city_delete]
            weights = np.delete(weights, city_delete)
            max_angles = np.delete(max_angles, city_delete)
            flag = False
        # Define the angular constraints for each city
        print(cities)
        print(max_angles)
        #initial_guess = np.mean(cities, axis=0)
        initial_guess = weiszfeld_algorithm(cities, weights)
        constraints = [
            {
                'type': 'ineq',
                'fun': angle_constraint,
                'args': (city, max_angle)
            }
            for city, max_angle in zip(cities, max_angles)
        ]

        
        # Perform the constrained optimization
        result = minimize(
            objective_function,
            initial_guess,
            
            args=(cities, weights),
            constraints=constraints,
            tol=tolerance,
            options={'maxiter': max_iterations}, 
            method = "trust-constr"
        )
        # Check if the optimization was successful
        if not result.success:
            logging.info("Optimization did not converge successfully.")
            logging.info(f"Message:{result.message}" )
        lowest_constraints = np.zeros(len(cities))
    
        # Check if the constraints are respected
        for i, constraint in enumerate(constraints):
            constraint_value = constraint['fun'](result.x, *constraint['args'])
            if constraint_value < -1:
                flag = True
                logging.info(f"Constraint {i} violated: value = {constraint_value}")
            else:
                logging.info(f"Constraint {i} respected: value = {constraint_value}")
            lowest_constraints[i] = constraint_value


        if not flag:
            break
        
        logging.info("Lagrange multipliers:", result.jac)
        # Return the optimized coordinates
    
    return result.x, cities_index
  '''  

def weiszfeld_algorithm(cities, weights, max_iterations=100, tolerance=1e-5):
    """
    Compute the geometric median using Weiszfeld's Algorithm.
    
    :param cities: List of (x, y) tuples representing city coordinates.
    :param weights: List of weights for each city.
    :param max_iterations: Maximum number of iterations.
    :param tolerance: Convergence tolerance.
    :return: Estimated geometric median (x, y).
    """
    # Initialize with the centroid as the starting point
    x_current, y_current = np.mean(cities, axis=0)
    
    for _ in range(max_iterations):
        numerator_x = 0
        numerator_y = 0
        denominator = 0
        
        for i, (x_i, y_i) in enumerate(cities):
            dist = np.sqrt((x_current - x_i) ** 2 + (y_current - y_i) ** 2)
            
            if dist != 0:  # To avoid division by zero
                weight = weights[i] / dist
                numerator_x += weight * x_i
                numerator_y += weight * y_i
                denominator += weight
        
        # Update the current estimate
        x_new = numerator_x / denominator
        y_new = numerator_y / denominator
        
        # Check for convergence
        if np.sqrt((x_new - x_current) ** 2 + (y_new - y_current) ** 2) < tolerance:
            return x_new, y_new
        
        x_current, y_current = x_new, y_new
    
    return x_current, y_current

def lat_lon_to_cartesian(lat, lon):
    """
    Convert latitude and longitude to 3D Cartesian coordinates.
    
    :param lat: Latitude in degrees.
    :param lon: Longitude in degrees.
    :return: Tuple (x, y, z) Cartesian coordinates.
    """
    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    
    # Earth's radius (mean radius)
    R = 6371.0  # in kilometers
    
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    logging.info(x, y, z)
    return x, y, z

def cartesian_to_lat_lon(x, y, z):
    """
    Convert 3D Cartesian coordinates to latitude and longitude.
    
    :param x: X coordinate.
    :param y: Y coordinate.
    :param z: Z coordinate.
    :return: Tuple (lat, lon) in degrees.
    """
    R = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / R)
    lon = np.arctan2(y, x)
    
    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    
    return lat, lon



def make_LOS_angles_df(choosen_areas_centroid , haps_position  ): 
        def calculate_los_angle(haps_position, centroid_pos):
            delta_x = np.abs(centroid_pos.x - haps_position.x)
            delta_y = np.abs(centroid_pos.y - haps_position.y)
            delta = np.sqrt(delta_x**2 + delta_y**2)
            
            angle_rad = np.arctan2(delta, 20000)  # Angle in radians
            angle_deg = np.degrees(angle_rad)  # Convert to degrees
            
            # Ensure angle is within 0 to 90 degrees
            #angle_deg = np.abs(angle_deg)
                
            return angle_deg
        

        angles = []
        
        for _, centroid_row in choosen_areas_centroid.iterrows():
            logging.info(centroid_row['name'])
            centroid_pos = centroid_row.geometry
            angle = calculate_los_angle(haps_position, centroid_pos)
            angles.append({
                 'Area_ID': centroid_row.ID,
                'Centroid_Name': centroid_row['name'],  # Use centroid name
                'Angle_Deg': angle
            })

        # Convert angles to DataFrame
        angles_df = pd.DataFrame(angles)
        return angles_df


def convert_to_Point(x, y):
    return Point(x, y)
