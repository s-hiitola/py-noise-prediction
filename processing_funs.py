import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString




def split_by_num_points(row, point_gdf):
    if not np.isnan(row.num_points) :
        points = int(row.num_points)
        if points > 1 and row.oneway == True:
            splitter = shapely.geometry.MultiPoint([row.geometry.interpolate((i/4), normalized=True) for i in range(1, points)])
            split_line = shapely.ops.split(shapely.ops.snap(row.geometry, splitter, 0.1), splitter)
            for segment in split_line:
                if point_gdf[point_gdf['tmsNumber']==row.tmsNumber].intersects(segment).any():
                    return segment
        else:
            return row.geometry


def road_is_not_link(row):
    if type(row['highway']) == list:
        for road_type in row['highway']:
            if 'link' in road_type:
                return False
    elif 'link' in row['highway']:
        return False
    return True


def capture_roads_by_names(row, roadnames):
    flattened = []
    for r in roadnames:
        if str(r) == 'nan':
            continue
        if type(r) == list:
            for sub_r in r:
                flattened.append(sub_r)
        else:
            flattened.append(r)
    return any(roadname in str(row['name']) for roadname in flattened)


def merge_multilines(row):
    if row['geometry'].type == "MultiLineString":
        single_line = shapely.ops.linemerge(row['geometry'])
        return single_line
    return row['geometry']


def combine_line_rows_to_one(row):
    return shapely.geometry.MultiLineString([row.parallel1,row.parallel2,row.parallel3,row.parallel4,row.parallel5,row.parallel6,row.parallel7,row.parallel8,row.parallel9,row.parallel10,row.parallel11,row.parallel12])


def add_z_coord_as_height_point(point):
    z = get_elevation_at_lat_lon('mosaic.tif', point.x, point.y)
    if z == None:
        z = 10
    
    return shapely.geometry.Point(point.x, point.y, z)

def combine_point_rows_to_one(row):
    multipoints = [row.points1,row.points2,row.points3,row.points4,row.points5,row.points6,row.points7,row.points8,row.points9,row.points10,row.points11,row.points12]
    points = []
    for multipoint in multipoints:
        for point in multipoint:
            points.append(add_z_coord_as_height_point(point))
    return shapely.geometry.MultiPoint(points)


def spread_points_on_lines(line, distance_delta):
    distances = np.arange(0, line.length, distance_delta)
    points = [line.interpolate(distance) for distance in distances] + [line.boundary[1]]
    multipoint = shapely.ops.unary_union(points)  # or new_line = LineString(points)
    return multipoint

def height_from_coords(x_coord, y_coord, bbox, heightmaps):
    #for x, y in zip(xs, ys):
    within_box = None
    for square, coords in bbox.items():
        x1,y1,x2,y2 = coords
        if x_coord > x1 and x_coord < x2 and y_coord > y1 and y_coord < y2:
            within_box = square
    if not within_box:
        return
    row, col = rasterio.transform.rowcol(heightmaps[within_box].transform, x_coord, y_coord)
    height_value = heightmaps[within_box].read(1)[row][col]
    #print(height_value)
    return height_value

def get_elevation_at_lat_lon(tif_path, lon, lat):
    with rasterio.open(tif_path) as dataset:
        # Convert lat/lon to pixel coordinates
        row, col = dataset.index(lon, lat)
        
        # Read the elevation value at the pixel coordinate
        elevation = dataset.read(1, window=((row, row+1), (col, col+1)))
        
        # Return the elevation value
        return elevation[0, 0]


def add_z_coord_as_height(row):
#     z =  height_from_coords(row.geometry.x, row.geometry.y)
    z =  get_elevation_at_lat_lon('mosaic.tif', row.geometry.x, row.geometry.y)
    #print(shapely.geometry.Point(row.geometry.x, row.geometry.y, z))
    #return shapely.geometry.Point(row.geometry.x, row.geometry.y)
    if z == None:
        z = 10
    
    return shapely.geometry.Point(row.geometry.x, row.geometry.y, z)


def read_csv_for_traffic(url):
    try:
        traffic_cols = ["tms_id","year","doy","h","m","s","ms","pituus (m)","kaista","suunta","class","nopeus (km/h)","faulty","kokonaisaika (tekninen)","aikaväli (tekninen)","jonoalku (tekninen)"]
        return pd.read_csv(url, names=traffic_cols, sep=";")
    except:
        print(url)
        return None

def my_interpolate(input_line, input_dist, normalized=False):
    '''
    From: https://stackoverflow.com/a/69489292
    
    Function that interpolates the coordinates of a shapely LineString.
    Note: If you use this function on a MultiLineString geometry, it will 
    "flatten" the geometry and consider all the points in it to be 
    consecutively connected. For example, consider the following shape: 
        MultiLineString(((0,0),(0,2)),((0,4),(0,6)))
    In this case, this function will assume that there is no gap between
    (0,2) and (0,4). Instead, the function will assume that these points
    all connected. Explicitly, the MultiLineString above will be 
    interpreted instead as the following shape:
        LineString((0,0),(0,2),(0,4),(0,6))

    Parameters
    ----------
    input_line : shapely.geometry.Linestring or shapely.geometry.MultiLineString
        (Multi)LineString whose coordinates you want to interpolate
    input_dist : float
        Distance used to calculate the interpolation point
    normalized : boolean
        Flag that indicates whether or not the `input_dist` argument should be
        interpreted as being an absolute number or a percentage that is 
        relative to the total distance or not.
        When this flag is set to "False", the `input_dist` argument is assumed 
        to be an actual absolute distance from the starting point of the 
        geometry. When this flag is set to "True", the `input_dist` argument 
        is assumed to represent the relative distance with respect to the 
        geometry's full distance.
        The default is False.

    Returns
    -------
    shapely.geometry.Point
        The shapely geometry of the interpolated Point.

    '''
    # Making sure the entry value is a LineString or MultiLineString
    if ((input_line.type.lower() != 'linestring') and 
        (input_line.type.lower() != 'multilinestring')):
        return None
    
    # Extracting the coordinates from the geometry
    if input_line.type.lower()[:len('multi')] == 'multi':
        # In case it's a multilinestring, this step "flattens" the points
        coords = [item for sub_list in [list(this_geom.coords) for 
                                        this_geom in input_line.geoms] 
                  for item in sub_list]
    else:
        coords = [tuple(coord) for coord in list(input_line.coords)]
    
    # Transforming the list of coordinates into a numpy array for 
    # ease of manipulation
    coords = np.array(coords)
    
    # Calculating the distances between points
    dists = ((coords[:-1] - coords[1:])**2).sum(axis=1)**0.5
    
    # Calculating the cumulative distances
    dists_cum = np.append(0,dists.cumsum())
    
    # Finding the total distance
    dist_total = dists_cum[-1]
    
    # Finding appropriate use of the `input_dist` value
    if normalized == False:
        input_dist_abs = input_dist
        input_dist_rel = input_dist / dist_total
    else:
        input_dist_abs = input_dist * dist_total
        input_dist_rel = input_dist
    
    # Taking care of some edge cases
    if ((input_dist_rel < 0) or 
        (input_dist_rel > 1) or 
        (input_dist_abs < 0) or 
        (input_dist_abs > dist_total)):
        return None
    elif ((input_dist_rel == 0) or (input_dist_abs == 0)):
        return shapely.geometry.Point(coords[0])
    elif ((input_dist_rel == 1) or (input_dist_abs == dist_total)):
        return shapely.geometry.Point(coords[-1])
    
    # Finding which point is immediately before and after the input distance
    pt_before_idx = np.arange(dists_cum.shape[0])[(dists_cum <= input_dist_abs)].max()
    pt_after_idx  = np.arange(dists_cum.shape[0])[(dists_cum >= input_dist_abs)].min()
    
    pt_before = coords[pt_before_idx]
    pt_after = coords[pt_after_idx]
    seg_full_dist = dists[pt_before_idx]
    dist_left = input_dist_abs - dists_cum[pt_before_idx]
    
    # Calculating the interpolated coordinates
    interpolated_coords = ((dist_left / seg_full_dist) * (pt_after - pt_before)) + pt_before
    
    # Creating a shapely geometry
    interpolated_point = shapely.geometry.point.Point(interpolated_coords)
    
    return interpolated_point

def split_linestrings(gdf, sub_length=10.0):
    # create empty lists to store new linestrings and indices
    new_linestrings = []
    indices = []
    sub_indices = []
    data = []
    
    # iterate over each linestring in the GeoDataFrame
    for i, row in gdf.iterrows():
        # Roads in tunnels are split in two, so that there is an object for both ends of the tunnel
        if row['tunnel'] == 'yes':
            new_coords = []
            new_coords.append(row.geometry.coords[0])
            point = my_interpolate(row.geometry, 0.5, normalized=True)
            new_coords.append(point.coords[0])
            new_coords.append(row.geometry.coords[1])
            point_pairs = zip(new_coords[:-1], new_coords[1:])
        
        # If the road is not in a tunnel, interpolate points along the linestring at equal intervals
        else:
            new_coords = []
            distance = 0.0
            while distance <= row.geometry.length:
                point = my_interpolate(row.geometry, distance)
                new_coords.append(point.coords[0])
                distance += sub_length
            new_coords.append(my_interpolate(row.geometry, row.geometry.length).coords[0]) # add the last point

            # create pairs of consecutive points
            point_pairs = zip(new_coords[:-1], new_coords[1:])

        # create a new linestring with each pair of points
        for j, pair in enumerate(point_pairs):
            new_linestring = shapely.geometry.linestring.LineString(pair)

            # add the new linestring and its indices to the lists
            new_linestrings.append(new_linestring)
            indices.append(i)
            sub_indices.append(j)
            data.append(row)

    # create a new GeoDataFrame with the new linestrings and the original data   
    
    new_gdf = gpd.GeoDataFrame({'geometry': new_linestrings, 'id': indices, 'sub_id': sub_indices}, crs='epsg:3067')
    for column in gdf.columns:
        if column not in ['geometry']:
            new_gdf[column] = [data[i][column] for i in range(len(new_gdf['id']))]
    return new_gdf


def straighten_road_elevation(roads):
    # Due to the low resolution of the elevation map, some of the road elevations may be incorrect. This should straighten the road.
    # Do not use in current form
    new_geoms = []
    for i, row in roads.iterrows():
        if row['tunnel'] == 'yes':
            new_geoms.append(row['geometry'])
            continue
        endpoint_1, endpoint_2 = row['geometry'].coords
        endpoint_1_el = get_elevation_at_lat_lon('mosaic.tif', endpoint_1[0], endpoint_1[1])
        endpoint_2_el = get_elevation_at_lat_lon('mosaic.tif', endpoint_2[0], endpoint_2[1])
        if row['highway'] == 'motorway' and new_geoms[i-1].coords[1][2] - endpoint_1_el > 1:
            endpoint_1_el = new_geoms[i-1].coords[1][2] - 0.02
        if row['highway'] == 'motorway' and endpoint_1_el - endpoint_2_el > 1.2:
            new_endpoint_1 = Point([endpoint_1[0], endpoint_1[1], new_geoms[i-1].coords[1][2] - 0.01])
            new_endpoint_2 = Point([endpoint_2[0], endpoint_2[1], new_geoms[i-1].coords[1][2] - 0.02])
            new_geoms.append(LineString([new_endpoint_1, new_endpoint_2]))
        elif row['highway'] == 'motorway' and endpoint_2_el - endpoint_1_el > 1.2:
            new_endpoint_1 = Point([endpoint_1[0], endpoint_1[1], new_geoms[i-1].coords[1][2] + 0.01])
            new_endpoint_2 = Point([endpoint_2[0], endpoint_2[1], new_geoms[i-1].coords[1][2] + 0.02])
            new_geoms.append(LineString([new_endpoint_1, new_endpoint_2]))
        elif row['highway'] == 'trunk' and endpoint_1_el - endpoint_2_el > 2:
            new_endpoint_1 = Point([endpoint_1[0], endpoint_1[1], new_geoms[i-1].coords[1][2] - 0.01])
            new_endpoint_2 = Point([endpoint_2[0], endpoint_2[1], new_geoms[i-1].coords[1][2] - 0.02])
            new_geoms.append(LineString([new_endpoint_1, new_endpoint_2]))
        elif row['highway'] == 'trunk' and endpoint_2_el - endpoint_1_el > 2:
            new_endpoint_1 = Point([endpoint_1[0], endpoint_1[1], new_geoms[i-1].coords[1][2] + 0.01])
            new_endpoint_2 = Point([endpoint_2[0], endpoint_2[1], new_geoms[i-1].coords[1][2] + 0.02])
            new_geoms.append(LineString([new_endpoint_1, new_endpoint_2]))
        else:
            new_geoms.append(row['geometry'])

    return new_geoms

def calculate_gradient(row):
    coords = row['geometry'].coords
    z1, z2 = coords[0][2], coords[-1][2]
    
    # calculate the horizontal distance between the points
    dx = row['geometry'].length
    # calculate the difference in elevation between the points
    dz = z2 - z1
    
    gradient = dz / dx
    while row['highway'] == 'motorway' and np.abs(gradient) > 0.05:
        gradient = gradient/2
    
    return np.abs(gradient)

def get_elevations_for_points(points_list):
    elevations = np.zeros(len(points_list), dtype='float64')
    with rasterio.open('mosaic.tif') as dataset:
        for i, point in enumerate(points_list):
            row, col = dataset.index(point.x, point.y)

            # Read the elevation value at the pixel coordinate
            elevation = dataset.read(1, window=((row, row+1), (col, col+1)))
            elevations[i] = elevation
    return elevations