import requests
import json
import pandas as pd
import geopandas as gpd
import io
import contextily as cx
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import osmnx as ox
import shapely
import numpy as np
import requests
import tifffile as tiff
import io
import rasterio
from rasterio.plot import show
import datetime as dt
from processing_funs import *
from noise_funs import *
from buildings import get_building_data_tampere

from fmiopendata.wfs import download_stored_query

plt.rcParams["figure.figsize"] = (50, 30)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


stations = requests.get("https://tie.digitraffic.fi//api/tms/v1/stations")
stations_t = json.load(io.StringIO(stations.text))
weather = requests.get("https://tie.digitraffic.fi/api/weather/v1/stations")
weather_t = json.load(io.StringIO(weather.text))
gdf_weather = gpd.GeoDataFrame.from_features(weather_t["features"])
gdf = gpd.GeoDataFrame.from_features(stations_t["features"])
gdf.crs = "EPSG:4326"

tampere = pd.concat([gdf.loc[["tampere" in c.lower() for c in  list(gdf['name'])]], gdf.loc[["tre" in c.lower() for c in  list(gdf['name'])]], gdf.loc[["rautaharkko" in c.lower() for c in  list(gdf['name'])]]])
tampere = tampere[tampere.name != "vt7_Treksilä"]
tampere = tampere[tampere.name != "vt3_Tampere_Myllypuro"]
tampere = tampere.to_crs(4326)

# Get the road network from openstreetmaps.
# Most of the steps concerning the street network would probably not be needed,
# if doing noise analysis for the entire network.
# Just don't simplify the graph and you should have the network in sections already,
# after turning the graph to gdf.
# Doing this twice and extracting the tunnels should allow for a similar
# handling of tunnel openings as the one used in the thesis, even if the rest of the network is different.


# First road network is for only getting the tunnels
G = ox.graph_from_place("Tampere, Finland", custom_filter='["highway"~"primary|secondary|motorway|trunk"]', simplify=False)
G = ox.simplification.simplify_graph(G, strict=False)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
gdf_edges = gdf_edges.to_crs(3067)

tunnels = gdf_edges[gdf_edges["tunnel"]=="yes"]

# Second network is for the rest of the roads
G = ox.graph_from_place("Tampere, Finland", custom_filter='["highway"~"primary|secondary|motorway|trunk"]', simplify=False)
G = ox.simplification.simplify_graph(G)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
gdf_edges = gdf_edges.to_crs(3067)

gdf_edges = gdf_edges[gdf_edges.apply(road_is_not_link, axis=1)]

tunnels = tunnels.reset_index(level=[0,1], drop=True).reset_index(drop=True)
simple_edges = gdf_edges.reset_index(level=[0,1], drop=True)
simple_edges['tunnel'] = 'null'


tampere = tampere.to_crs(3067)
tampere['buffer'] = tampere['geometry'].buffer(5)

tampere = tampere.set_geometry('buffer')

new_gdf = gpd.sjoin(simple_edges[['name','oneway','geometry','highway','tunnel']], tampere[['buffer','tmsNumber']], predicate='intersects')

new_gdf = new_gdf.reset_index(drop=True)

only_roads_with_stations = simple_edges[simple_edges.apply(capture_roads_by_names, args=([i for i in new_gdf['name']], ), axis=1) | simple_edges['name'].isnull()]
only_roads_with_stations_tunnels = gpd.overlay(only_roads_with_stations[['name','oneway','geometry','highway']], tunnels[['tunnel','geometry']], how='union')
only_roads_with_stations_tunnels = only_roads_with_stations_tunnels.join(
    gpd.sjoin(tampere, only_roads_with_stations_tunnels).groupby("index_right").size().rename("num_points"),
    how="left",
)

joined_station_in_tunnel = gpd.sjoin(only_roads_with_stations_tunnels, tampere[['buffer','tmsNumber']], op='intersects')
joined_station_in_tunnel['geometry'] = joined_station_in_tunnel.apply(merge_multilines, axis=1)

new_gdf = new_gdf.set_index('index_right', drop=True)
new_gdf.index.name = None

# Split road if there are multiple points on it
new_gdf = new_gdf.join(
    gpd.sjoin(tampere, new_gdf).groupby("index_right").size().rename("num_points"),
    how="left",
)
new_gdf.geometry = new_gdf.apply(split_by_num_points, axis=1, args=(tampere,))


test_buffer = new_gdf.copy()
test_buffer['geometry'] = new_gdf.buffer(40, single_sided=True)
only_roads_with_stations_tunnels.index.name = None
test = gpd.clip(test_buffer, only_roads_with_stations_tunnels[only_roads_with_stations_tunnels['tunnel'].isnull()])
both_lanes = gpd.overlay(test, tunnels[tunnels['name']=='Tampereen itäinen kehätie'], how='union')

just_one_tunnel = joined_station_in_tunnel[joined_station_in_tunnel['tunnel'] == 'yes']
new_gdf = new_gdf.overlay(just_one_tunnel, how='identity')
new_gdf['tunnel'] = new_gdf.apply(lambda row: "yes" if row['tunnel_2'] == 'yes' or row['tunnel_1'] == 'yes' else np.nan, axis=1)
new_gdf = new_gdf.rename(columns={'num_points_1':'num_points','name_1':'name','oneway_1':'oneway','highway_1':'highway','tmsNumber_1':'tmsNumber'})
new_gdf = new_gdf.drop(['name_2','oneway_2','tunnel_1','tunnel_2','highway_2','tmsNumber_2'],axis=1)
new_gdf.explore()

new_buffer = gpd.GeoDataFrame(geometry=new_gdf.buffer(50, single_sided=True), crs='epsg:3067')
other_buffer = gpd.GeoDataFrame(geometry=new_gdf.buffer(5, single_sided=True), crs='epsg:3067')
gdf_one_lane = both_lanes.clip(new_buffer.overlay(other_buffer, how='difference'), keep_geom_type=True)


gdf_filter = gdf_one_lane.copy()

for i, row in gdf_one_lane.iterrows():
    if type(row.geometry) == shapely.geometry.collection.GeometryCollection:
        lines = []
        # Make two lines from the parts of lines, keep the longer one
        for shape in row.geometry:
            if type(shape) == shapely.geometry.linestring.LineString:
                lines.append(shape)
        merged = shapely.ops.linemerge(lines)
        gdf_filter.at[i, 'geometry'] = merged

tampere['buffer'] = tampere.buffer(165)
tampere = tampere.set_geometry('buffer')
gdf_filtered = gpd.sjoin(gdf_filter, tampere, op='intersects').set_geometry('geometry_left')
gdf_filtered.index.name = None
gdf_filtered['geometry'] = gdf_filtered['geometry_left']
gdf_filtered = gdf_filtered.set_geometry('geometry')


gdf_combined = new_gdf.append(gdf_filtered)
#gdf_combined = gdf_combined.dissolve(by='id')
# gdf_combined.shape
gdf_combined = gdf_combined[gdf_combined.geom_type != 'Point']
gdf_combined.explore()

gdf_combined['tunnel'] = gdf_combined.apply(lambda row: "yes" if row['tunnel_2'] == 'yes' or row['tunnel_1'] == 'yes' or row['tunnel'] == 'yes' else np.nan, axis=1)
gdf_combined['oneway'] = gdf_combined.apply(lambda row: "yes" if row['oneway_2'] == 'yes' or row['oneway_1'] == 'yes' or row['oneway'] == 'yes' else np.nan, axis=1)
gdf_combined['highway'] = gdf_combined.apply(lambda row: row['highway_1'] if type(row['highway_1'])==str or type(row['highway_1'])==list else (row['highway_2'] if type(row['highway_2'])==str or type(row['highway_2'])==list else row['highway']), axis=1)
gdf_combined['tmsNumber'] = gdf_combined.apply(lambda row: row['tmsNumber'] if ~np.isnan(row['tmsNumber']) else (row['tmsNumber_left'] if ~np.isnan(row['tmsNumber_left']) else row['tmsNumber_right']), axis=1)
gdf_combined['name'] = gdf_combined.apply(lambda row: row['name_1'] if type(row['name_1'])==str else (row['name_2'] if type(row['name_2'])==str else row['name']), axis=1)
gdf_combined = gdf_combined[['name','oneway','geometry','highway','tunnel','tmsNumber']]

gdf_combined = gdf_combined.reset_index(drop=True)
gdf_combined = gdf_combined.explode()

gdf_combined['parallel1'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(10, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel2'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(15, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel3'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(21, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel4'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(28, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel5'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(37, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel6'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(48, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel7'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(62, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel8'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(78, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel9'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(98, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel10'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(120, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel11'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(146, side="right", resolution=16, join_style=2, mitre_limit=10))
gdf_combined['parallel12'] = gdf_combined['geometry'].apply(lambda row: row.parallel_offset(174, side="right", resolution=16, join_style=2, mitre_limit=10))

gdf_combined['points1'] = gdf_combined['parallel1'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points2'] = gdf_combined['parallel2'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points3'] = gdf_combined['parallel3'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points4'] = gdf_combined['parallel4'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points5'] = gdf_combined['parallel5'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points6'] = gdf_combined['parallel6'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points7'] = gdf_combined['parallel7'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points8'] = gdf_combined['parallel8'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points9'] = gdf_combined['parallel9'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points10'] = gdf_combined['parallel10'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points11'] = gdf_combined['parallel11'].apply(spread_points_on_lines, args=(15,))
gdf_combined['points12'] = gdf_combined['parallel12'].apply(spread_points_on_lines, args=(15,))


# Remember to change the function back
gdf_combined['all_points'] = gdf_combined.apply(combine_point_rows_to_one, axis=1)


gdf_combined.head()
point_cloud = gdf_combined.set_geometry('all_points')

tmsNumbers = []
points = []

for index, row in point_cloud.iterrows():
    tmsNumber = row['tmsNumber']
    for point in row['all_points']:
        points.append(point)
        tmsNumbers.append(tmsNumber)
            
calculation_points = gpd.GeoDataFrame({"tmsnumber": tmsNumbers, "geometry": points}, geometry=points, crs="epsg:3067")


tam_buildings = get_building_data_tampere(gdf_combined)

# Add a marker for points within buildings
# Buffer by 1 to prevent touching
buildings_buffered = gpd.GeoDataFrame(geometry=tam_buildings.buffer(1), crs="epsg:3067")

points_in_buildings = gpd.sjoin(calculation_points, buildings_buffered, op='intersects')

# filter out the points that have an intersection with the polygons
calculation_points['in_building'] = calculation_points.index.isin(points_in_buildings.index)

reindexed = []
for multi_idx, row in gdf_combined.iterrows():
    i, j = multi_idx
    new_row = {}
    if j != 0:
        continue
    reindexed.append(row)
reindexed_gdf = gpd.GeoDataFrame(data=reindexed, geometry='geometry', crs='epsg:3067').reset_index(drop=True)
reindexed_gdf['tmsNumber'] = reindexed_gdf['tmsNumber'].astype(int)
road_gdf = reindexed_gdf[['name','tunnel','tmsNumber','highway','geometry']]

road_gdf = gdf_combined[['name','tunnel','tmsNumber','highway','geometry']]

grouped = road_gdf.groupby(level=0)['geometry'].apply(lambda x: shapely.ops.linemerge(x.unary_union) if len(x) > 1 else x.iloc[0]).reset_index(drop=True)
# Create a new geodataframe with the merged lines and other columns from the original geodataframe
new_gdf = gpd.GeoDataFrame(geometry=grouped, crs=road_gdf.crs)
new_gdf[['name','tunnel','highway','tmsNumber']] = road_gdf[['name','tunnel','highway','tmsNumber']].groupby(level=0).first()

road_gdf = new_gdf.explode().reset_index(drop=True)

tmsNums_with_tunnels = road_gdf[road_gdf['tunnel'] == 'yes']['tmsNumber'].astype(int).values.tolist()
tunnel_buffers_elevations = {}

road_idx_and_geom = []

for idx, row in road_gdf.iterrows():
    if row['tunnel'] != 'yes':
        new_line = shapely.geometry.linestring.LineString([(coord[0], coord[1], get_elevation_at_lat_lon('mosaic.tif',coord[0], coord[1])) for coord in row['geometry'].coords])
        row['geometry'] = new_line
        if row['tmsNumber'] in tmsNums_with_tunnels:
            # Get endpoints of roads that have tunnels but are not in the tunnel.
            for endpoint in row['geometry'].boundary:
                # Test if the endpoint is actually in the tunnel already, thus giving an incorrect elevation value
                # Create a  point 3 meters down the road from the endpoint
                test = shapely.geometry.linestring.LineString(list(endpoint.buffer(3).exterior.coords)).intersection(new_line)
                # Find the elevation at the point
                elevation_3_meters_down_the_road = get_elevation_at_lat_lon('mosaic.tif', test.x, test.y)
                # If
                if endpoint.z > elevation_3_meters_down_the_road+1.5:
                    elevation = elevation_3_meters_down_the_road
                    
                    # Create a replacement of the road that changes the elevation to the level at which the road is actually
                    replacement = []
                    for i, point in enumerate(new_line.coords):
                        if (i+1 == 1 or i+1 ==len(new_line.coords)) and point[2] == endpoint.z:
#                             print(point)
#                             print(elevation_3_meters_down_the_road)
#                             print(endpoint)
                            replacement.append([point[0], point[1], elevation])
                            #continue
                        replacement.append([point[0], point[1], point[2]])
                    row['geometry'] = shapely.geometry.linestring.LineString(replacement)
                else:
                    elevation = endpoint.z
                
                tunnel_buffers_elevations.setdefault('buffer', []).append(endpoint.buffer(10))
                tunnel_buffers_elevations.setdefault('z', []).append(elevation)
        # Save the geometry data and index here since it doesn't work otherwise
        road_idx_and_geom.append((idx, row['geometry']))

idx_and_endpoint_elevations_for_tunnels = []
for idx, row in road_gdf.iterrows():
    if row['tunnel'] == 'yes':
        print(idx)
        endpoint1, endpoint2 = row['geometry'].boundary
        for i, buf in enumerate(tunnel_buffers_elevations['buffer']):
            if endpoint1.within(buf):
                endpoint1_elev = tunnel_buffers_elevations['z'][i]
            if endpoint2.within(buf):
                endpoint2_elev = tunnel_buffers_elevations['z'][i]
        idx_and_endpoint_elevations_for_tunnels.append((idx, shapely.geometry.linestring.LineString([[endpoint1.x, endpoint1.y, endpoint1_elev], [endpoint2.x, endpoint2.y, endpoint2_elev]])))


for idx, geom in road_idx_and_geom:
    road_gdf.loc[idx, 'geometry'] = geom

for idx, geom in idx_and_endpoint_elevations_for_tunnels:
    road_gdf.loc[idx, 'tunnel'] = 'yes'
    road_gdf.loc[idx, 'geometry'] = geom

for index, road in road_gdf.iterrows():
    if road['highway'] == 'motorway':
        road_gdf.loc[index, 'hard_surface'] = 7.75
    else:
        shortest_distance = float('inf')
        for idx, polygon in tam_buildings.iterrows():
            distance = road['geometry'].distance(polygon.geometry)
            if distance < shortest_distance:
                shortest_distance = distance
        road_gdf.loc[index, 'hard_surface'] = shortest_distance

roads = split_linestrings(road_gdf, sub_length=25)

roads['gradient'] = roads.apply(calculate_gradient, axis=1)
roads = roads.rename(columns={'tmsNumber': 'tmsnumber'})

# Currently, all points lie on the surface of the terrain, add these to z coordinates to lift them from the ground
receiver_height = 2
source_height = 0.5
neighboring = True

point_ids = []
road_ids = []
road_subs = []
delta_L_rs = []
delta_L_MSs = []
delta_L_AVs = []
delta_L_alphas = []

np.seterr(all='raise')

for i_point, row in calculation_points.iterrows():
    receiver_point = row['geometry']
    receiver_point = Point(receiver_point.x, receiver_point.y, receiver_point.z+receiver_height)
    
    
    distances = roads[roads['tmsnumber']==row['tmsnumber']].distance(receiver_point)
    closest_segment = roads.loc[distances.idxmin()]
    if neighboring:
        segments = find_neighboring_sections(roads, closest_segment)
#         else:
#             try:
#                 prev_segment = roads[roads.intersects(Point(closest_segment.geometry.coords[0]).buffer(5)) & roads['id'] != closest_segment['id']].iloc[0]
#                 next_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']+1)].iloc[0]
#                 segments = [closest_segment, next_segment]
#             except IndexError:
#                 segments = [closest_segment]

    else:
        segments = [closest_segment]

        
    wall_normals = []
    alpha_rs = []
    lines_to_corners_list = []
    
    
    i_buildings = []
    

    for road_segment in segments:
        
        if not row['in_building']:
        
            dist_point_road = road_segment['geometry'].distance(receiver_point)  # 2D Distance
            road_segment_linestring = road_segment['geometry']
            road_segment_midpoint = my_interpolate(road_segment_linestring, 0.5, normalized=True)
            if not road_segment['tunnel'] == True:
                road_segment_midpoint = Point(road_segment_midpoint.x, road_segment_midpoint.y, get_elevation_at_lat_lon('mosaic.tif', road_segment_midpoint.x, road_segment_midpoint.y))
            road_endpoint_1, road_endpoint_2 = road_segment_linestring.coords
            line_point_road_1 = LineString([road_endpoint_1, receiver_point])
            line_point_road_2 = LineString([road_endpoint_2, receiver_point])
            alpha_road = calculate_angle_between_lines(line_point_road_1, line_point_road_2, True)            
            delta_L_alpha = 10*np.log10(alpha_road/np.pi)            
            
            # Set the "midpoint" to the mouth of the tunnel if the road in question is in a tunnel
            # In the case of tunnels, the mouth of the tunnel is the source
            if road_segment['tunnel'] == True:
                alpha_road = 0.1
                # Each tunnel is made up of two segments, 0 and 1
                if road_segment['sub_index'] == 0:
                    road_segment_midpoint = road_segment_linestring.coords[0]
                else:
                    road_segment_midpoint = road_segment_linestring.coords[1]

            # Create a point that is 'source_height' above the midpoint of the road
            source_point = Point(road_segment_midpoint.x, road_segment_midpoint.y, road_segment_midpoint.z+source_height)

            connection_source_receiver = LineString([source_point, receiver_point])
            dist_source_receiver = calculate_3D_length(connection_source_receiver)  # 3D distance
            
            if dist_source_receiver < 20:
                building_search_distance = 20
            else:
                building_search_distance = int(dist_source_receiver)

            buildings_subset = get_subset_of_buildings(tam_buildings, connection_source_receiver, building_search_distance)

            reflection_plane_s, reflection_plane_r, max_indices, elevations, o1_on_plane, o2_on_plane, h_e, h_m, h_b, h_v = find_barriers(connection_source_receiver, road_segment_midpoint, buildings_subset)

            if not (max_indices[0] == 0 and max_indices[1] == 0):
                d_1 = np.sqrt((o1_on_plane.x - source_point.x) ** 2 + (o1_on_plane.y - source_point.y) ** 2 + (o1_on_plane.z - source_point.z) ** 2)
                d_2 = np.sqrt((receiver_point.x - o2_on_plane.x) ** 2 + (receiver_point.y - o2_on_plane.y) ** 2 + (receiver_point.z - o2_on_plane.z) ** 2)
            else:
                d_1 = calculate_3D_length(reflection_plane_r)
                d_2 = d_1

            if d_1 == 0:
                d_1 = 1
            if d_2 == 0:
                d_1 = 1


            if max_indices[0] == 0 and max_indices[1] == 0:
                screened = False
#                 print(road_segment['hard_surface'])
#                 print(d_1)
                if d_1 > road_segment['hard_surface']:
                    ground_type_hard = False
                else:
                    ground_type_hard = True
#                 ground_type_hard = False
                delta_L_MS = calculate_delta_L_M(d_1, h_m, h_b, ground_type_hard)
            else:
                screened = True
                delta_L_screen = calculate_delta_L_screen(dist_source_receiver, d_1, d_2, h_e, h_m, h_b, h_v, max_indices)
                # Test what the ground type is behind the screen/barrier
                if max_indices[1]*2 > road_segment['hard_surface']:
                    ground_type_hard = False
                else:
                    ground_type_hard = True
                delta_L_m, z = calculate_delta_L_m(d_2, h_m, h_b, h_v, delta_L_screen, ground_type_hard)
                if np.around(d_1) > np.around(road_segment['hard_surface']):
                    ground_type_hard = False
                else:
                    ground_type_hard = True
                delta_L_VS = calculate_delta_L_M(d_1, h_m, h_b, ground_type_hard)
                delta_L_MS = delta_L_screen + delta_L_VS*(1-((7*z)/13)) + delta_L_m

            delta_L_AV = -10*np.log10((np.sqrt(dist_point_road**2 + (h_m - h_b - 0.5)**2))/10)

            # Calculate the angle between the receiver and the normal of the road
            road_normal = calculate_normal(road_segment_linestring, receiver_point)
            angle = calculate_angle_between_normal_and_point(road_normal, receiver_point)
    #         print(np.degrees(angle))


            delta_L_reflections = [0]

            normal_gdf = gpd.GeoDataFrame(geometry=[road_normal], crs="epsg:3067")

            for i_building, building_row in buildings_subset.iterrows():
                reflecting, alpha_reflection_radians, perspective_center, lines_to_corners, wall_normal = deal_with_a_building(road_normal, building_row, receiver_point, road_segment_linestring, dist_point_road, buildings_subset, False)
                if reflecting:
                    wall_normals.append(wall_normal)
                    alpha_rs.append(np.degrees(alpha_reflection_radians))
                    i_buildings.append(i_building)
                    lines_to_corners_list.append(lines_to_corners)
                    # Determine the distance from the road to the "image receiver"
                    dist_road_reflection_shortest = closest_segment['geometry'].distance(perspective_center)
                    dist_road_receiver_shortest = closest_segment['geometry'].distance(receiver_point)
                    if dist_road_reflection_shortest > dist_road_receiver_shortest:
                        dist_road_image_receiver = dist_road_receiver_shortest + (dist_road_reflection_shortest-dist_road_receiver_shortest)*2
                    else:
                        dist_road_image_receiver = dist_road_receiver_shortest

                    delta_L_reflection = 10*np.log10(1+((alpha_reflection_radians*dist_road_receiver_shortest)/(alpha_road*dist_road_image_receiver)))
                    delta_L_reflections.append(delta_L_reflection)
        
        if row['in_building']:
            delta_L_rs.append(0)
            delta_L_MSs.append(0)
            delta_L_AVs.append(0)
            delta_L_alphas.append(0)

        else:
            sum_of_delta_L_reflections = sum(delta_L_reflections)
            delta_L_rs.append(sum_of_delta_L_reflections)
            delta_L_MSs.append(delta_L_MS)
            delta_L_AVs.append(delta_L_AV)
            delta_L_alphas.append(delta_L_alpha)
        point_ids.append(i_point)
        road_ids.append(road_segment['id'])
        road_subs.append(road_segment['sub_id'])

delta_data = {
    'point_id': point_ids,
    'road_id': road_ids,
    'road_sub': road_subs,
    'delta_L_r': delta_L_rs,
    'delta_L_MS': delta_L_MSs,
    'delta_L_AV': delta_L_AVs,
    'delta_L_alpha': delta_L_alphas,
}


delta_df = pd.DataFrame(delta_data)
delta_df.to_csv('delta_data.csv')