import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box, Point
from shapely.ops import unary_union
import numpy as np
import pandas as pd 
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


class Map(object): 
    def __init__(self, Map_grid : tuple = (150, 150), haps_grid_positions : tuple = (20,20), corners : tuple = (9.02, 45.35, 9.32, 45.615)): 
        self.MapGrid = Map_grid
        self.HapsPoisitionsGrid = haps_grid_positions
        self.min_lat =  corners[0]
        self.min_lon = corners[1]
        self.max_lat = corners[2]
        self.max_lon = corners[3]

    def create_milan_coordinates(self): 

     
        bbox = gpd.GeoDataFrame({'geometry': [box(self.min_lat, self.min_lon, self.max_lat, self.max_lon)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)
        
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], self.MapGrid[0] + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], self.MapGrid[1] + 1)

        #Create the grid cells as polygons
        return x_coords, y_coords , bbox
    
    
    
    
    
    def Areas_Positions(self): 
        
        # Provided dictionary with area names and types
        area_dict = {
    'Agricultural Park': 'suburban_rural',
    'Business': 'dense_urban',
    'Duomo': 'dense_urban',
    'FS': 'dense_urban',
    'Highway': 'suburban_rural',
    'Industrial': 'urban',
    'Linate': 'urban',
    'Mediolanum Forum': 'urban',
    'Monza Park': 'suburban_rural',
    'Polimi': 'urban',
    'Residential 1': 'urban',
    'Residential 2': 'urban',
    'Rho': 'urban',
    'San Siro': 'urban',
    'Theatre': 'urban',
    'Tourism Attraction': 'dense_urban'
        }

# Extract area types from the dictionary (order matters here)
        max_angles = list(area_dict.values())
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        # Create a grid of 150x150 within the bounding box
        rows = cols = 150
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create a grid of 50x50
    def Areas_Positions(self): 
        
        # Provided dictionary with area names and types
        area_dict = {
    'Agricultural Park': 'suburban_rural',
    'Business': 'dense_urban',
    'Duomo': 'dense_urban',
    'FS': 'dense_urban',
    'Highway': 'suburban_rural',
    'Industrial': 'urban',
    'Linate': 'urban',
    'Mediolanum Forum': 'urban',
    'Monza Park': 'suburban_rural',
    'Polimi': 'urban',
    'Residential 1': 'urban',
    'Residential 2': 'urban',
    'Rho': 'urban',
    'San Siro': 'urban',
    'Theatre': 'urban',
    'Tourism Attraction': 'dense_urban'
        }

# Extract area types from the dictionary (order matters here)
        max_angles = list(area_dict.values())
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        # Create a grid of 150x150 within the bounding box
        rows = cols = 150
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create a grid of 50x50
    def Areas_Positions(self): 
        
        # Provided dictionary with area names and types
        area_dict = {
    'Agricultural Park': 'suburban_rural',
    'Business': 'dense_urban',
    'Duomo': 'dense_urban',
    'FS': 'dense_urban',
    'Highway': 'suburban_rural',
    'Industrial': 'urban',
    'Linate': 'urban',
    'Mediolanum Forum': 'urban',
    'Monza Park': 'suburban_rural',
    'Polimi': 'urban',
    'Residential 1': 'urban',
    'Residential 2': 'urban',
    'Rho': 'urban',
    'San Siro': 'urban',
    'Theatre': 'urban',
    'Tourism Attraction': 'dense_urban'
        }

# Extract area types from the dictionary (order matters here)
        max_angles = list(area_dict.values())
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        # Create a grid of 150x150 within the bounding box
        rows = cols = 150
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create a grid of 50x50
    def Areas_Positions(self): 
        
        # Provided dictionary with area names and types
        area_dict = {
    'Agricultural Park': 'suburban_rural',
    'Business': 'dense_urban',
    'Duomo': 'dense_urban',
    'FS': 'dense_urban',
    'Highway': 'suburban_rural',
    'Industrial': 'urban',
    'Linate': 'urban',
    'Mediolanum Forum': 'urban',
    'Monza Park': 'suburban_rural',
    'Polimi': 'urban',
    'Residential 1': 'urban',
    'Residential 2': 'urban',
    'Rho': 'urban',
    'San Siro': 'urban',
    'Theatre': 'urban',
    'Tourism Attraction': 'dense_urban'
        }

# Extract area types from the dictionary (order matters here)
        max_angles = list(area_dict.values())
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        # Create a grid of 150x150 within the bounding box
        rows = cols = 150
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create a grid of 50x50
    def Areas_Positions(self): 
        
        # Provided dictionary with area names and types
        area_dict = {
    'Agricultural Park': 'suburban_rural',
    'Business': 'dense_urban',
    'Duomo': 'dense_urban',
    'FS': 'dense_urban',
    'Highway': 'suburban_rural',
    'Industrial': 'urban',
    'Linate': 'urban',
    'Mediolanum Forum': 'urban',
    'Monza Park': 'suburban_rural',
    'Polimi': 'urban',
    'Residential 1': 'urban',
    'Residential 2': 'urban',
    'Rho': 'urban',
    'San Siro': 'urban',
    'Theatre': 'urban',
    'Tourism Attraction': 'dense_urban'
        }

# Extract area types from the dictionary (order matters here)
        max_angles = list(area_dict.values())
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        # Create a grid of 150x150 within the bounding box
        rows = cols = 150
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create a grid of 50x50
    
    def Areas_Positions(self): 
        
        # Provided dictionary with area names and types
        area_dict = {
    'Agricultural Park': 'suburban_rural',
    'Business': 'dense_urban',
    'Duomo': 'dense_urban',
    'FS': 'dense_urban',
    'Highway': 'suburban_rural',
    'Industrial': 'urban',
    'Linate': 'urban',
    'Mediolanum Forum': 'urban',
    'Monza Park': 'suburban_rural',
    'Polimi': 'urban',
    'Residential 1': 'urban',
    'Residential 2': 'urban',
    'Rho': 'urban',
    'San Siro': 'urban',
    'Theatre': 'urban',
    'Tourism Attraction': 'dense_urban'
        }
        traffic_load = {
    'Theatre': 61.176659,
    'Duomo': 44.011478,
    'Business': 42.908182,
    'Agricultural Park': 41.128871,
    'FS': 33.127922,
    'Tourism Attraction': 33.005956,
    'Residential 2': 28.017033,
    'Linate': 21.403835,
    'Polimi': 18.446788,
    'Monza Park': 14.588566,
    'Industrial': 11.295296,
    'Highway': 10.642662,
    'Rho': 6.345979,
    'San Siro': 4.421755,
    'Mediolanum Forum': 4.136733,
    'Residential 1': 1.302191
}
# Extract area types from the dictionary (order matters here)
        max_angles = list(area_dict.values())
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        # Create a grid of 150x150 within the bounding box
        rows = cols = 150
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create a grid of 50x50
        small_rows = small_cols = 20
        x_small_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], small_cols + 1)
        y_small_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], small_rows + 1)

        # Create the 150x150 grid cells as polygons
        grid_cells = [box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
                    for i in range(cols) for j in range(rows)]
        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs='EPSG:3857')
        
        grid_centroids = np.array([(cell.centroid.x, cell.centroid.y) for cell in grid.geometry])

        # Apply k-means clustering
        num_areas = 16  # Adjust the number of clusters as needed
        kmeans = KMeans(n_clusters=num_areas, random_state=1998).fit(grid_centroids)
        labels = kmeans.labels_

        # Assign clusters to zones
        grid['area'] = labels
        clustered_areas = grid.dissolve(by='area')

        # Compute centroids for each clustered area
        clustered_areas['centroid'] = clustered_areas.geometry.centroid
        # Assign initial labels to areas (for simplicity, directly based on order)
        # Add an ID column for the areas
        np.random.seed(1999)
        clustered_areas = clustered_areas.sample(frac=1).reset_index(drop=True)
        sorted_areas = sorted(traffic_load.items(), key=lambda x: -x[1])

# Get centroids of all areas
        centroids = np.array([(geom.x, geom.y) for geom in clustered_areas['centroid']])

        # Initialize assignment
        assigned_indices = []
        assigned_names = []
        remaining_indices = list(range(len(centroids)))
        area_names = list(area_dict.keys())
        # Assign areas based on traffic load and maximizing distance
        for name, load in sorted_areas:
            if not assigned_indices:
                # Assign the first area to the centroid that is farthest from the center (or arbitrarily)
                assigned_idx = remaining_indices.pop(0)
                assigned_indices.append(assigned_idx)
                assigned_names.append(name)
            else:
                # Remaining centroids and their indices
                rem_centroids = centroids[remaining_indices]
                # Compute distances from remaining centroids to already assigned centroids
                distances = cdist(rem_centroids, centroids[assigned_indices], metric='euclidean')
                # Sum distances to all assigned centroids
                sum_distances = distances.sum(axis=1)
                # Find the centroid with the maximum sum distance
                farthest_idx_in_rem = np.argmax(sum_distances)
                assigned_idx = remaining_indices.pop(farthest_idx_in_rem)
                assigned_indices.append(assigned_idx)
                assigned_names.append(name)

        # Now, assign the area names to the corresponding regions in clustered_areas
        clustered_areas = clustered_areas.copy()
        clustered_areas['area_name'] = ''
        clustered_areas['area_type'] = ''
        clustered_areas['traffic_load'] = 0.0

        for idx, name in zip(assigned_indices, assigned_names):
            clustered_areas.at[idx, 'area_name'] = name
            clustered_areas.at[idx, 'area_type'] = area_dict[name]
            clustered_areas.at[idx, 'traffic_load'] = traffic_load[name]

        # Compute adjacency relationships
  

        # Initialize the adjacency matrix
        num_areas = len(clustered_areas)
        area_ordered = list(area_dict.keys())
## order the areas in the clustered_areas
        clustered_areas = clustered_areas.set_index('area_name').loc[area_ordered].reset_index()
        print(clustered_areas)
        return clustered_areas, bbox.total_bounds
    def Areas_Positions_1(self): 
        
        # Provided dictionary with area names and types
        area_dict = {
    'Agricultural Park': 'suburban_rural',
    'Business': 'dense_urban',
    'Duomo': 'dense_urban',
    'FS': 'dense_urban',
    'Highway': 'suburban_rural',
    'Industrial': 'urban',
    'Linate': 'urban',
    'Mediolanum Forum': 'urban',
    'Monza Park': 'suburban_rural',
    'Polimi': 'urban',
    'Residential 1': 'urban',
    'Residential 2': 'urban',
    'Rho': 'urban',
    'San Siro': 'urban',
    'Theatre': 'urban',
    'Tourism Attraction': 'dense_urban'
        }

# Extract area types from the dictionary (order matters here)
        max_angles = list(area_dict.values())
        minx, miny = 9.02, 45.35  # Southwest corner
        maxx, maxy = 9.32, 45.615  # Northeast corner

        # Create a GeoDataFrame with a single rectangle (bounding box) in EPSG:4326
        bbox = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        # Create a grid of 150x150 within the bounding box
        rows = cols = 150
        x_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create a grid of 50x50
        small_rows = small_cols = 20
        x_small_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], small_cols + 1)
        y_small_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], small_rows + 1)

        # Create the 150x150 grid cells as polygons
        grid_cells = [box(x_coords[i], y_coords[j], x_coords[i + 1], y_coords[j + 1])
                    for i in range(cols) for j in range(rows)]
        grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs='EPSG:3857')
        
        grid_centroids = np.array([(cell.centroid.x, cell.centroid.y) for cell in grid.geometry])

        # Apply k-means clustering
        num_areas = 16  # Adjust the number of clusters as needed
        kmeans = KMeans(n_clusters=num_areas, random_state=1998).fit(grid_centroids)
        labels = kmeans.labels_

        # Assign clusters to zones
        grid['area'] = labels
        clustered_areas = grid.dissolve(by='area')

        # Compute centroids for each clustered area
        clustered_areas['centroid'] = clustered_areas.geometry.centroid
        # Assign initial labels to areas (for simplicity, directly based on order)
        # Add an ID column for the areas
        clustered_areas['area_id'] = range(len(clustered_areas))

        # Compute adjacency relationships
        adjacency = clustered_areas.geometry.apply(lambda geom: clustered_areas.geometry.touches(geom))

        # Initialize the adjacency matrix
        num_areas = len(clustered_areas)
        adj_matrix = np.zeros((num_areas, num_areas), dtype=int)

        # Populate the adjacency matrix
        for i in range(num_areas):
            for j in range(num_areas):
                if i != j and adjacency.iloc[i][j]:
                    adj_matrix[i, j] = 1

        # Convert adjacency matrix to a DataFrame for better readability
        adj_matrix_df = pd.DataFrame(
            adj_matrix, 
            index=clustered_areas['area_id'], 
            columns=clustered_areas['area_id']
        )

        # Display the adjacency matrix
        print("Adjacency Matrix:")
        print(adj_matrix_df)
        clustered_areas['area_type'] = max_angles

        # Enforce adjacency constraints (simple heuristic)
        # Iterate through adjacency matrix to find areas with different types
        #adj_matrix_df = pd.DataFrame(adjacency_matrix, index=clustered_areas.index, columns=clustered_areas.index)

        for i, row in adj_matrix_df.iterrows():
            neighbors = row[row == 1].index  # Get indices of adjacent areas
            for neighbor in neighbors:
                if clustered_areas.loc[i, 'area_type'] == clustered_areas.loc[neighbor, 'area_type']:
                    # Skip if swapping violates the type counts
                    continue

        # Map area names strictly to their corresponding types from the dictionary
        area_names = []
        remaining_areas = area_dict.copy()  # Create a mutable copy of the dictionary

        for area_type in clustered_areas['area_type']:
            # Match area type to a name from the dictionary
            name = next((k for k, v in remaining_areas.items() if v == area_type), None)
            if name:
                area_names.append(name)
                del remaining_areas[name]  # Remove assigned name to prevent reuse
            else:
                area_names.append("Unassigned")  # Fallback in case of mismatches

        # Add the names to the GeoDataFrame
        clustered_areas['area_name'] = area_names
        print(clustered_areas)
        return clustered_areas, bbox.total_bounds
    def Areas_Positions_1(self): 
   
        areas = areas = {'Agricultural Park': {'x': range(51, 10, -1), 'y': range(26, 5, -1), 'color': '#6B8E23'},
          'Business': {'x': range(83, 86), 'y': range(65, 68), 'color': '#FFFF00'}, 
          'Duomo': {'x': [83, 84], 'y': [63, 62], 'color': '#FFC0CB'}, 
          'FS': {'x': range(91, 97), 'y': range(75, 83), 'color': '#800080'},
            'Highway': {'x': range(72, 52, -1), 'y': range(21, 28), 'color': '#008080'} ,
              'Highway 2': { 'x': list(range(44, 53)), 'y': list(range(27, 34)),  'color': '#8B008B'  # Dark Magenta
    },
      'Industrial': {'x': range(100, 115), 'y': range(54, 61), 'color': '#A52A2A'},
        'Linate': {'x': range(123, 131), 'y': range(52, 62), 'color': '#0000FF'}, 
        'Mediolanum Forum': {'x': range(54, 63), 'y': range(29, 38), 'color': '#C0C0C0'},
          'Monza Park': {'x': range(124, 141), 'y': range(133, 149), 'color': '#006400'},
            'Polimi': {'x': range(100, 110), 'y': range(68, 76), 'color': '#FFA500'},
              'Residential 1': {'x': range(66, 76), 'y': range(50, 53), 'color': '#808080'}, 
              'Residential 2': {'x': range(86, 93), 'y': range(87, 93), 'color': '#87CEFA'}, 
              'Rho': {'x': range(7, 14), 'y': range(99, 103), 'color': '#FF0000'}, 
              'San Siro': {'x': range(50, 61), 'y': range(71, 77), 'color': '#00FFFF'},
                'Theatre': {'x': range(95, 102), 'y': range(89, 97), 'color': '#483D8B'},
                  'Tourism Attraction': {'x': range(81, 76, -1), 'y': range(65, 72), 'color': '#32CD32'}}


        x_coords, y_coords, bbox = self.create_milan_coordinates()
        # Create GeoDataFrames for each specified area
        cols, rows = self.MapGrid
        area_gdfs = {}
        for name, info in areas.items():
            geometries = []
            for x in info['x']:
                for y in info['y']:
                    if 0 <= x < cols and 0 <= y < rows:
                        geometries.append(box(x_coords[x], y_coords[y], x_coords[x + 1], y_coords[y + 1]))
            
            area_gdfs[name] = gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:3857')  
        

        merged_gdfs = {}
        centroids = {}
        for name, info in areas.items():
            if name in ['Highway', 'Highway 2']:
                if 'Highway' not in merged_gdfs:
                    merged_gdfs['Highway'] = gpd.GeoDataFrame({'geometry': []}, crs='EPSG:3857')
                
                # Combine geometries for 'Highway' and 'Highway 2'
                geometries = area_gdfs[name]['geometry']
                merged_gdfs['Highway'] = pd.concat([merged_gdfs['Highway'], gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:3857')])
                
                # Calculate centroid of merged areas
                merged_geometry = unary_union(merged_gdfs['Highway']['geometry'])
                centroid = merged_geometry.centroid
                centroids['Highway'] = centroid
                centroids['Highway 2'] = centroid
            else:
                geometries = area_gdfs[name]['geometry']
                merged_gdfs[name] = gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:3857')
                
                # Calculate centroid of non-merged areas
                merged_geometry = unary_union(merged_gdfs[name]['geometry'])
                centroid = merged_geometry.centroid
                centroids[name] = centroid 


        self.clustered_areas = gpd.GeoDataFrame(
                        {'geometry': [centroids[name] for name in areas.keys()]},
                        crs='EPSG:3857'
                        )
        # delete highway 2
        self.clustered_areas = self.clustered_areas.drop(5)
        del areas['Highway 2']
        self.clustered_areas['name'] = list(areas.keys())
        self.clustered_areas.reset_index(drop=True, inplace=True)
        self.clustered_areas["ID"] = self.clustered_areas.index
        return self.clustered_areas, bbox.total_bounds
    

    def Haps_Positions(self): 
        bbox = gpd.GeoDataFrame({'geometry': [box(self.min_lat, self.min_lon, self.max_lat, self.max_lon)]}, crs='EPSG:4326')

        # Reproject the bounding box to Web Mercator (EPSG:3857) for use with contextily
        bbox = bbox.to_crs(epsg=3857)

        cols, rows = self.Haps_PositionsGrid
        haps_positions = []
        x_haps_coords = np.linspace(bbox.total_bounds[0], bbox.total_bounds[2], cols + 1)
        y_haps_coords = np.linspace(bbox.total_bounds[1], bbox.total_bounds[3], rows + 1)

        # Create the grid cells as polygons
        haps_pos = [Point((x_haps_coords[i] + x_haps_coords[i + 1]) / 2,
                            (y_haps_coords[j] + y_haps_coords[j + 1]) / 2)
                        for i in range(cols) for j in range(rows)]
        self.haps_pos_gdf = gpd.GeoDataFrame({'geometry': haps_pos}, crs='EPSG:3857')
        
    def make_LOS_angles_df(self): 
        def calculate_los_angle(drone_pos, centroid_pos):
            delta_x = np.abs(centroid_pos.x - drone_pos.x)
            delta_y = np.abs(centroid_pos.y - drone_pos.y)
            delta = np.sqrt(delta_x**2 + delta_y**2)
            
            angle_rad = np.arctan2(delta, 20000)  # Angle in radians
            angle_deg = np.degrees(angle_rad)  # Convert to degrees
            
            # Ensure angle is within 0 to 90 degrees
            angle_deg = np.abs(angle_deg)
                
            return angle_deg
        

        angles = []
        for _, drone_row in self.haps_pos_gdf.iterrows():
            drone_pos = drone_row.geometry
            for _, centroid_row in self.areas_centroid_gdf.iterrows():
                print(centroid_row['name'])
                centroid_pos = centroid_row.geometry
                angle = calculate_los_angle(drone_pos, centroid_pos)
                angles.append({
                    'Drone_X': drone_pos.x,
                    'Drone_Y': drone_pos.y,
                    'Centroid_Name': centroid_row['name'],  # Use centroid name
                    'Angle_Deg': angle
                })

        # Convert angles to DataFrame
        angles_df = pd.DataFrame(angles)

        # Save the DataFrame to a CSV file
        angles_df.to_csv('angles_from_drone_to_centroids.csv', index=False) 





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