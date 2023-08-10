from shapely.geometry.linestring import LineString
from shapely.geometry.point import Point
from processing_funs import *
import numpy as np
import math

def find_barriers(connection_source_receiver, road_segment_midpoint, buildings_subset, cnossos=False, reflection=False):
    # Create a set of points along the connecting line at 2 meter intervals
    # This will basically convert the coordinates of the points from x,y,z space to u,v space
    # In the u,v space, the x (u) coordinate represents both the original x and y coordinates as the distance along the line,
    # while the y (v) coordinate represents the z coordinate of the x,y,z space.
    
    
    interval_points = []
    for distance_delta in range(0, int(connection_source_receiver.length)+1, 2):
        point_along_line = my_interpolate(connection_source_receiver, distance_delta)
        interval_points.append(point_along_line)

    # Get the elevation values at each point on the terrain surface
    elevations = get_elevations_for_points(interval_points)

    # Get elevations along the straight line connecting the source and receiver
    connecting_line_elevations = np.array([point.z for point in interval_points])

    h_e, h_vm = [0, 0]
    o1_on_plane, o2_on_plane = [None, None]
    
    buildings_intersect = buildings_subset[buildings_subset.geometry.intersects(connection_source_receiver)]
    # In the case that there are buildings, the building heights need to be added to the elevation values
    if not buildings_intersect.empty and not reflection:
        
        # There will now be two receivers and two sources
        # barrier_receiver is the point on the edge of the barrier on the side of the source i.e. the road   
        for barrier, barrier_elevation in zip(buildings_intersect['geometry'], buildings_intersect['roof_elevation']):
            barriers = barrier.boundary.intersection(connection_source_receiver)
            barrier_point_pairs = zip(barriers[:-1:2], barriers[1::2])
            for barrier_receiver, barrier_source in barrier_point_pairs:                
                barrier_receiver = Point(barrier_receiver.x, barrier_receiver.y, barrier_elevation)
                barrier_source = Point(barrier_source.x, barrier_source.y, barrier_elevation)
                
                start_index = int(np.floor(barrier_receiver.distance(road_segment_midpoint) / 2))
                stop_index = int(np.floor(barrier_source.distance(road_segment_midpoint) / 2))
                
                # In some cases the indexes will be in the wrong order. Correct if necessary
                if start_index > stop_index:
                    temp = stop_index
                    stop_index = start_index
                    start_index = temp
                # Add the roof elevation of the intersecting building to the terrain elevations
                elevations[start_index:stop_index+1] = barrier_elevation

    # Find the effective heights of points between the real source and receiver
    elevation_difference = elevations-connecting_line_elevations    
    # There is a barrier if the terrain rises 1.5 meters above the sight line
    if np.around(np.max(elevation_difference),1) >= 0.0 and not reflection:
        max_indices = get_max_indices(elevation_difference)
        if max_indices[0] == 0:
            reflection_plane_source_side, slope_source_side, intercept_source_side = determine_reflection_plane(connection_source_receiver.coords[0], interval_points[max_indices[0]].coords[0], [elevations[max_indices[0]]])
        elif max_indices[0] == 1:
            reflection_plane_source_side, slope_source_side, intercept_source_side = determine_reflection_plane(connection_source_receiver.coords[0], interval_points[max_indices[0]].coords[0], elevations[:max_indices[0]])
        else:
            reflection_plane_source_side, slope_source_side, intercept_source_side = determine_reflection_plane(connection_source_receiver.coords[0], interval_points[max_indices[0]-1].coords[0], elevations[:max_indices[0]-1])
        if max_indices[1]+1 >= len(interval_points):
            reflection_plane_receiver_side, slope_receiver_side, intercept_receiver_side = determine_reflection_plane(interval_points[max_indices[1]].coords[0], interval_points[-1].coords[0], elevations[max_indices[1]:])
        else:
            reflection_plane_receiver_side, slope_receiver_side, intercept_receiver_side = determine_reflection_plane(interval_points[max_indices[1]+1].coords[0], interval_points[-1].coords[0], elevations[max_indices[1]+1:])
    
    
        # Locations of the source and receiver side of the barrier
        u_proj_o1 = (max_indices[0] + slope_source_side * elevations[max_indices[0]] - slope_source_side * intercept_source_side) / (1 + slope_source_side**2)
        v_proj_o1 = (slope_source_side * max_indices[0] + slope_source_side**2 * elevations[max_indices[0]] + intercept_source_side) / (1 + slope_source_side**2)

        u_proj_o2 = (max_indices[1] + slope_receiver_side * elevations[max_indices[1]] - slope_receiver_side * intercept_receiver_side) / (1 + slope_receiver_side**2)
        v_proj_o2 = (slope_receiver_side * max_indices[1] + slope_receiver_side**2 * elevations[max_indices[1]] + intercept_receiver_side) / (1 + slope_receiver_side**2)

        if u_proj_o1 < 0:
            u_proj_o1 = 0
        if round(u_proj_o1) >= len(interval_points):
            u_proj_o1 = (len(interval_points))-0.5
            
        if u_proj_o2 < 0:
            u_proj_o2 = 0
        if round(u_proj_o2) >= len(interval_points):
            u_proj_o2 = (len(interval_points))-0.5
            
        try:
            o1_on_plane = Point(my_interpolate(connection_source_receiver, u_proj_o1*2).x, my_interpolate(connection_source_receiver, u_proj_o1*2).y, v_proj_o1)
        except AttributeError:
            u_proj_o1 = (calculate_3D_length(connection_source_receiver)/2)-1
            o1_on_plane = Point(my_interpolate(connection_source_receiver, u_proj_o1*2).x, my_interpolate(connection_source_receiver, u_proj_o1*2).y, v_proj_o1)
        try:
            o2_on_plane = Point(my_interpolate(connection_source_receiver, u_proj_o2*2).x, my_interpolate(connection_source_receiver, u_proj_o2*2).y, v_proj_o2)
        except AttributeError:
            u_proj_o2 = (calculate_3D_length(connection_source_receiver)/2)-1
            o2_on_plane = Point(my_interpolate(connection_source_receiver, u_proj_o2*2).x, my_interpolate(connection_source_receiver, u_proj_o2*2).y, v_proj_o2)
        h_e = np.max(elevation_difference)
        h_vm = abs(slope_receiver_side*max_indices[1] - elevations[max_indices[1]] + intercept_receiver_side) / np.sqrt(slope_receiver_side**2 + 1)    
        h_vb = abs(slope_source_side*max_indices[0] - elevations[max_indices[0]] + intercept_source_side) / np.sqrt(slope_source_side**2 + 1)
    
        point_mid_barrier = interval_points[np.around(np.sum(max_indices)*0.5).astype('int')]

        midpoint_for_s = Point(point_mid_barrier.coords[0][0], point_mid_barrier.coords[0][1], elevations[max_indices[0]]-h_vb)
        midpoint_for_r = Point(point_mid_barrier.coords[0][0], point_mid_barrier.coords[0][1], elevations[max_indices[1]]-h_vm)
        reflection_plane_source_side = LineString([reflection_plane_source_side.coords[0], midpoint_for_s.coords[0]])
        reflection_plane_receiver_size = LineString([midpoint_for_r.coords[0], reflection_plane_receiver_side.coords[1]])
        
    else: # No building or terrain barrier
        max_indices = np.array([0,0])
        # Note that the source and receiver heights do not affect the reflection plane
        # Only the xy coordinates of source and receiver are used, elevations come from terrain elevation data    
        reflection_plane, slope, intercept = determine_reflection_plane(connection_source_receiver.coords[0], connection_source_receiver.coords[-1], elevations)
        reflection_plane_receiver_side = reflection_plane
        reflection_plane_source_side = reflection_plane
        slope_receiver_side = slope
        slope_source_side = slope
        intercept_receiver_side = intercept
        intercept_source_side = intercept
    
    

    if cnossos:
        h_e = np.max(elevation_difference)
        
        z_or = 0
        z_os = 0
        barrier_point_source = None
        barrier_point_middle = None
        barrier_point_receiver = None
        
        if max_indices[0] != 0 or max_indices[1] != 0:
            z_or = h_vm   
            z_os = h_vb

            point_source_corner = interval_points[max_indices[0]]
            point_receiver_corner = interval_points[max_indices[1]]

            barrier_point_source = Point(point_source_corner.coords[0][0], point_source_corner.coords[0][1], elevations[max_indices[0]])
            barrier_point_middle = Point(point_mid_barrier.coords[0][0], point_mid_barrier.coords[0][1], elevations[np.around(np.sum(max_indices)*0.5).astype('int')])
            barrier_point_receiver = Point(point_receiver_corner.coords[0][0], point_receiver_corner.coords[0][1], elevations[max_indices[1]])
        
        # Calculate the coordinates of the source and receiver points along the infinite line
        u_proj_s = (0 + slope_source_side * connecting_line_elevations[0] - slope_source_side * intercept_source_side) / (1 + slope_source_side**2)
        v_proj_s = (slope_source_side * 0 + slope_source_side**2 * connecting_line_elevations[0] + intercept_source_side) / (1 + slope_source_side**2)
        
        u_proj_r = (connection_source_receiver.length + slope_receiver_side * connecting_line_elevations[-1] - slope_receiver_side * intercept_receiver_side) / (1 + slope_receiver_side**2)
        v_proj_r = (slope_receiver_side * connection_source_receiver.length + slope_receiver_side**2 * connecting_line_elevations[-1] + intercept_receiver_side) / (1 + slope_receiver_side**2)        
        
        
        # Calculate the distance between the point and its projection onto the line
        z_s = np.sqrt((0 - u_proj_s)**2 + (connecting_line_elevations[0] - v_proj_s)**2)
        z_r = np.sqrt((connection_source_receiver.length - u_proj_r)**2 + (connecting_line_elevations[-1] - v_proj_r)**2)

        # Calculate the coordinates of the mirror images
        u_mirror_s = 2 * u_proj_s - 0
        v_mirror_s = 2 * v_proj_s - connecting_line_elevations[0]
        
        u_mirror_r = 2 * u_proj_r - connection_source_receiver.length
        v_mirror_r = 2 * v_proj_r - connecting_line_elevations[-1]
        
        if u_mirror_s < 0:
            u_mirror_s = 0
        if round(u_mirror_s) > len(interval_points):
            u_mirror_s = (len(interval_points))-0.5
            
        if u_mirror_r < 0:
            u_mirror_r = 0
        if round(u_mirror_r) > len(interval_points):
            u_mirror_r = (len(interval_points))-0.5
        

        s_dot = Point(my_interpolate(connection_source_receiver, u_mirror_s).x, my_interpolate(connection_source_receiver, u_mirror_s).y, v_mirror_s)
        r_dot = Point(my_interpolate(connection_source_receiver, u_mirror_r).x, my_interpolate(connection_source_receiver, u_mirror_r).y, v_mirror_r)
        
        if connection_source_receiver.coords[0][2] < reflection_plane_source_side.coords[0][2]:
            z_s = -z_s
        if connection_source_receiver.coords[1][2] < reflection_plane_receiver_side.coords[1][2]:
            z_r = -z_r
        
        return reflection_plane_source_side, reflection_plane_receiver_side, s_dot, r_dot, barrier_point_source, barrier_point_receiver, o1_on_plane, o2_on_plane, max_indices, elevations, h_e, z_s, z_r, z_os, z_or
    
    h_m = abs(slope_receiver_side*connection_source_receiver.length - connection_source_receiver.coords[1][2] + intercept_receiver_side) / np.sqrt(slope_receiver_side**2 + 1)
    h_b = abs(slope_source_side*0 - connection_source_receiver.coords[0][2] + intercept_source_side) / np.sqrt(slope_source_side**2 + 1)
    if connection_source_receiver.coords[0][2] < reflection_plane_source_side.coords[0][2]:
        h_b = -h_b
    if connection_source_receiver.coords[1][2] < reflection_plane_receiver_side.coords[1][2]:
        h_m = -h_m
    
    return reflection_plane_source_side, reflection_plane_receiver_side, max_indices, elevations, o1_on_plane, o2_on_plane, h_e, h_m, h_b, h_vm

def calculate_delta_L_M(distance, h_m, h_b, hard):
    # Calculate ground correction
    
    if hard:
        return 0
    
    sigma = (distance*10**(-0.3*h_b))/(10*h_m)
#     print(sigma)
    if sigma > 1:
        return -6*np.log10((sigma**2)/1+0.01*sigma**2)
    else:
        return 0


def calculate_delta_L_m(d_2, h_m, h_b, h_v, delta_L_screen, hard):
    # Calculate ground correction in screened situations
    
    if delta_L_screen <= -18:
        z = 1
    elif delta_L_screen > 18 and delta_L_screen <= -5:
        z = (-delta_L_screen - 5)/13
    else:
        z = 0
    
    if hard:
        if h_m >= 2:
            sigma = d_2/(10*h_m)
        else:
            sigma = 0
        if sigma < 0.2:
            return 0, z
        elif sigma <= 10:
            return 3*z*np.log10(sigma) + 2*z, z
        else:
            return 5*z, z
    if h_v > 4:
        h_v = 4
    sigma = (d_2*10**(-0.3*h_v))/10*h_m
    
    if sigma <= 0.1:
        return 0, z
    if sigma <= 0.3:
        return 2*z - 4*z*np.log10(0.3/sigma), z
    if sigma < 1:
        return -4*z*np.log(sigma), z
    else:
        return -5*(1-z)*np.log10((sigma**2)/(1+0.01*sigma**2)), z
    
def calculate_delta_L_thick_screen(h_v, h_m, h_b, d, d_1, e):
    V_s = np.degrees(np.arctan((h_v-h_b-0.5)/d_1))
    V_m = np.degrees(np.arctan((h_v-h_m)/(d-d_1-e)))
    V_s = np.abs(V_s)
    V_m = np.abs(V_m)
    u = (3.5 + V_s + V_m - np.sqrt(100+V_s**2+V_m**2 - 1.6*V_s*V_m + 7*(V_s + V_m)))/18
    k = 11 - 10**((6-u)/6)
    delta_L_ts = -k*np.log10(2.2*(e-0.05))
    if delta_L_ts > 0:
        delta_L_ts = 0
    return delta_L_ts
    

def calculate_delta_L_screen(d, d_1, d_2, h_e, h_m, h_b, h_v, max_indices):
    if d_2 >= d_1:
        if d_2 < 30:
            d_2 = 30
        elif d_2/d_1 > 20:
            d_1 = 0.05
    else:
        if d_1 < 30:
            d_1 = 30
        elif d_1/d_2 > 20:
            d_2 = 0.05
    x = 1.1 * h_e * np.sqrt((d_2+d_1)/(d_2*d_1))
    if x >= 2.4:
        delta_L_screen = -25.0
    elif x >= 0:
        delta_L_screen = -5 - 10*np.log10(1 + x + 17*x**2)
    elif x >= -0.33:
        delta_L_screen = -5 + 10*np.log10(1 - x + 17*x**2)
    else:
        delta_L_screen = 0.0
        
    e = (max_indices[1]-max_indices[0])*2
    delta_L_ts = 0
    if e > 1:
        delta_L_ts = calculate_delta_L_thick_screen(h_v, h_m, h_b, d, d_1, e)
    delta_L_screen = delta_L_screen+delta_L_ts
    
    return delta_L_screen

def calculate_3D_length(line):
    x1, y1, z1 = line.coords[0]
    x2, y2, z2 = line.coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    return length

def find_neighboring_sections(roads, closest_segment):
    if closest_segment['sub_id'] > 1:
        prev_prev_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']-2)].iloc[0]
        prev_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']-1)].iloc[0]
        try:
            next_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']+1)].iloc[0]
            try:
                next_next_segment = roads[(roads['id']==next_segment['id'])&(roads['sub_id']==next_segment['sub_id']+1)].iloc[0]
                segments = [prev_prev_segment, prev_segment, closest_segment, next_segment, next_next_segment]
            
            except IndexError: # Last segment of the road
                try:
                    next_next_segment = roads[(roads.intersects(Point(next_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                    segments = [prev_prev_segment, prev_segment, closest_segment, next_segment, next_next_segment]
                
                except IndexError: # No segments left in the tmsnumber
                    segments = [prev_prev_segment, prev_segment, closest_segment, next_segment]
        
        except IndexError:
            try:
                next_segment = roads[(roads.intersects(Point(closest_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                try:
                    next_next_segment = roads[(roads['id']==next_segment['id'])&(roads['sub_id']==next_segment['sub_id']+1)].iloc[0]
                    segments = [prev_prev_segment, prev_segment, closest_segment, next_segment, next_next_segment]
                
                except IndexError:
                    segments = [prev_prev_segment, prev_segment, closest_segment, next_segment]
            
            except IndexError: # Last segment of the road
                segments = [prev_prev_segment, prev_segment, closest_segment]
            
            
    elif closest_segment['sub_id'] > 0:
        prev_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']-1)].iloc[0]
        try:
            prev_prev_segment = roads[(roads.intersects(Point(prev_segment.geometry.coords[0]).buffer(5))) & (roads['id'] != prev_segment['id'])].iloc[0]
            try:
                next_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']+1)].iloc[0]
                try:
                    next_next_segment = roads[(roads['id']==next_segment['id'])&(roads['sub_id']==next_segment['sub_id']+1)].iloc[0]
                    segments = [prev_prev_segment, prev_segment, closest_segment, next_segment, next_next_segment]
                
                except IndexError: # Last segment of the road
                    try:
                        next_next_segment = roads[(roads.intersects(Point(next_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                        segments = [prev_prev_segment, prev_segment, closest_segment, next_segment, next_next_segment]
                    
                    except IndexError: # No segments left in the tmsnumber
                        segments = [prev_prev_segment, prev_segment, closest_segment, next_segment]
            
            except IndexError:
                try:
                    next_segment = roads[(roads.intersects(Point(closest_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                    try:
                        next_next_segment = roads[(roads['id']==next_segment['id'])&(roads['sub_id']==next_segment['sub_id']+1)].iloc[0]
                        segments = [prev_prev_segment, prev_segment, closest_segment, next_segment, next_next_segment]
                    
                    except IndexError:
                        segments = [prev_prev_segment, prev_segment, closest_segment, next_segment]
                
                except IndexError: # Last segment of the road
                    segments = [prev_prev_segment, prev_segment, closest_segment]
            
        except IndexError:
            try:
                next_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']+1)].iloc[0]
                try:
                    next_next_segment = roads[(roads['id']==next_segment['id'])&(roads['sub_id']==next_segment['sub_id']+1)].iloc[0]
                    segments = [prev_segment, closest_segment, next_segment, next_next_segment]
                except IndexError: # Last segment of the road
                    try:
                        next_next_segment = roads[(roads.intersects(Point(next_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                        segments = [prev_segment, closest_segment, next_segment, next_next_segment]
                    except IndexError: # No segments left in the tmsnumber
                        segments = [prev_segment, closest_segment, next_segment]
            except IndexError:
                try:
                    next_segment = roads[(roads.intersects(Point(closest_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                    try:
                        next_next_segment = roads[(roads['id']==next_segment['id'])&(roads['sub_id']==next_segment['sub_id']+1)].iloc[0]
                        segments = [prev_segment, closest_segment, next_segment, next_next_segment]
                    except IndexError:
                        segments = [prev_segment, closest_segment, next_segment]
                except IndexError: # Last segment of the road
                    segments = [prev_segment, closest_segment]
            
    else: # Sub_id == 0
        try:
            prev_segment = roads[(roads.intersects(Point(closest_segment.geometry.coords[0]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
            prev_prev_segment = roads[(roads['id']==prev_segment['id'])&(roads['sub_id']==prev_segment['sub_id']-1)].iloc[0]
            next_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']+1)].iloc[0]
            try:
                next_next_segment = roads[(roads.intersects(Point(next_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                segments = [prev_segment, prev_prev_segment, closest_segment, next_segment, next_next_segment]
            except IndexError:
                segments = [prev_segment, prev_prev_segment, closest_segment, next_segment]
        except IndexError:
            next_segment = roads[(roads['id']==closest_segment['id'])&(roads['sub_id']==closest_segment['sub_id']+1)].iloc[0]
            try:
                next_next_segment = roads[(roads.intersects(Point(next_segment.geometry.coords[1]).buffer(5))) & (roads['id'] != closest_segment['id'])].iloc[0]
                segments = [closest_segment, next_segment, next_next_segment]
            except IndexError:
                segments = [closest_segment, next_segment]
    return segments

def reflection_surface(wall_and_road_facing_ea, surface, surface_normal, point, road_segment, buildings_subset):
    # Calculate the angle embodying the end points of the reflecting surface seen from the receiver.
    # Return the perspective middle point, the length to the perspective middle, and the perspective angle
    # 
    #              Point receiver
    #                /  /
    #              /   /
    #       Wall_1    /
    #          | \   /
    #    wall  |   \/
    #  surface |   / \ 
    #          |  /    \
    #       Wall_2       \
    #           \          \
    #            \           \
    #             \            \
    #              \             \
    #            Road_1-----------Road_2

    
    surface_1, surface_2 = surface.coords
    road_1, road_2 = road_segment.coords
    
    # Endpoint 1 should be the one closest to the receiver/hearer
    if point.distance(Point(surface_1)) > point.distance(Point(surface_2)):
        surface_endpoint1 = Point(surface_2)
        surface_endpoint2 = Point(surface_1)
        
    else:
        surface_endpoint1 = Point(surface_1)
        surface_endpoint2 = Point(surface_2)
        
    # Create lines from the point to the endpoints of the surface
    line_point_1 = LineString([surface_endpoint1, point])
    line_point_2 = LineString([surface_endpoint2, point])    
    
    
    # Check if the reflection lines are crossing buildings, if yes, stop calculation
    if buildings_subset.crosses(line_point_2).any() and buildings_subset.crosses(line_point_1).any():
        return False, 0, 0, 0, shapely.geometry.multilinestring.MultiLineString([line_point_1, line_point_2])
        
    # Set the road endpoint 1 to be the point closes to surface endpoint 2
    if surface_endpoint2.distance(Point(road_1)) > surface_endpoint2.distance(Point(road_2)):
        road_endpoint1 = Point(road_2)
        road_endpoint2 = Point(road_1)
        
    else:
        road_endpoint1 = Point(road_1)
        road_endpoint2 = Point(road_2)
    
    # Create lines from the endpoints of the road segment to the endpoints of the surface
    # The first number is the endpoint of the road, the second number is the point on the surface
    line_road_1_1 = LineString([surface_endpoint1, road_endpoint1])
    line_road_1_2 = LineString([surface_endpoint2, road_endpoint1])
    line_road_2_1 = LineString([surface_endpoint1, road_endpoint2])
    line_road_2_2 = LineString([surface_endpoint2, road_endpoint2])
    
    
    angle_normal_road_1_1 = calculate_angle_between_lines(line_road_1_1, surface_normal, True)
    angle_normal_road_2_1 = calculate_angle_between_lines(line_road_2_1, surface_normal, True)
    
    # If both endpoints of the road are behind the wall, exit the function
    if angle_normal_road_1_1 > 0.5*np.pi and angle_normal_road_2_1 > 0.5*np.pi:
        return False, 0, 0, 0, shapely.geometry.multilinestring.MultiLineString([line_point_1, line_point_2])
    
    # If one endpoint of the road is behind the building wall, move the endpoint to the center.
    while angle_normal_road_2_1 > 0.5*np.pi or angle_normal_road_1_1 > 0.5*np.pi:
        if angle_normal_road_2_1 > 0.5*np.pi:
            road_endpoint2 = my_interpolate(LineString([road_endpoint1, road_endpoint2]), 0.5, True)
            line_road_2_1 = LineString([surface_endpoint1, road_endpoint2])
            line_road_2_2 = LineString([surface_endpoint2, road_endpoint2])
            angle_normal_road_2_1 = calculate_angle_between_lines(line_road_2_1, surface_normal, True)
        elif angle_normal_road_1_1 > 0.5*np.pi:
            road_endpoint1 = my_interpolate(LineString([road_endpoint1, road_endpoint2]), 0.5, True)
            line_road_1_1 = LineString([surface_endpoint1, road_endpoint1])
            line_road_1_2 = LineString([surface_endpoint2, road_endpoint1])
            angle_normal_road_1_1 = calculate_angle_between_lines(line_road_1_1, surface_normal, True)
            
#     line_segments = []
#     for road_endpoint, surface_endpoint in [(road_endpoint1, surface_endpoint1), (road_endpoint1, surface_endpoint2), (road_endpoint2, surface_endpoint1), (road_endpoint2, surface_endpoint2)]:
#         line_segments.append(LineString([surface_endpoint, road_endpoint]))
        
#         # Calculate angle between endpoint and surface normal
#         angle_normal = calculate_angle_between_lines(line_segments[-1], surface_normal, True)
        
#         # Adjust endpoint if necessary
#         if angle_normal > 0.5*np.pi:
#             if road_endpoint == road_endpoint1:
#                 road_endpoint1 = my_interpolate(LineString([road_endpoint1, road_endpoint2]), 0.5, True)
#             else:
#                 road_endpoint2 = my_interpolate(LineString([road_endpoint1, road_endpoint2]), 0.5, True)
            
            
    if buildings_subset.crosses(line_road_1_1).any() and buildings_subset.crosses(line_road_2_2).any():
        return False, 0, 0, 0, shapely.geometry.multilinestring.MultiLineString([line_point_1, line_point_2])

    

    angle_normal_road_1_2 = calculate_angle_between_lines(line_road_1_2, surface_normal, True)
    angle_normal_road_2_2 = calculate_angle_between_lines(line_road_2_2, surface_normal, True)    
    angle_normal_point_1 = calculate_angle_between_lines(line_point_1, surface_normal, True)
    angle_normal_point_2 = calculate_angle_between_lines(line_point_2, surface_normal, True)
    
    angle_normal_point_1_0_2pi = calculate_angle_between_lines(line_point_1, surface_normal, False)
    angle_normal_point_2_0_2pi = calculate_angle_between_lines(line_point_2, surface_normal, False)
    
    facing_wall = False
    if (angle_normal_point_1_0_2pi > np.pi and angle_normal_point_2_0_2pi < np.pi) or (angle_normal_point_1_0_2pi < np.pi and angle_normal_point_2_0_2pi > np.pi):
        facing_wall = True

#     print('facing wall?', facing_wall)
    if not check_angles(wall_and_road_facing_ea, facing_wall, angle_normal_point_1, angle_normal_point_2, angle_normal_road_1_1, angle_normal_road_1_2, angle_normal_road_2_1, angle_normal_road_2_2):
        return False, 0, 0, 0, shapely.geometry.multilinestring.MultiLineString([line_point_1, line_point_2])
    
    # Calculate the angle between lines connecting the point to the two endpoints
    perspective_angle = calculate_angle_between_lines(line_point_2, line_point_1, True)
    middle_angle = perspective_angle / 2

    # Calculate the length of the line from the point to the perspective middle point
    perspective_distance = abs(surface.length / (2 * math.sin(middle_angle)))

    # Calculate the coordinates of the perspective middle point
    middle_x = point.x + perspective_distance * np.sin(middle_angle)
    middle_y = point.y + perspective_distance * np.cos(middle_angle)

    # Return the perspective middle point as a Shapely Point object
    perspective_midpoint = Point([middle_x, middle_y])        

    return True, perspective_angle, perspective_distance, perspective_midpoint, shapely.geometry.multilinestring.MultiLineString([line_point_1, line_point_2])


def deal_with_a_building(road_normal, building, receiver_point, road_segment, dist_point_road, buildings_subset, print_angles):
    road_segment_midpoint = my_interpolate(road_segment, 0.5, True)
    polygon = building.geometry
    distance_receiver_building = receiver_point.distance(polygon)
    distance_source_building = road_segment_midpoint.distance(polygon)
    
    # Get the walls of the building
    boundary = polygon.exterior

    # Find each wall
    coords = list(boundary.coords)
    
    
    
    for i in range(len(coords) - 1):
        wall = LineString([coords[i], coords[i+1]])
        # Skip very short sections of wall
        if wall.length < 2:
            continue
        midpoint = my_interpolate(wall, 0.5, normalized=True)
        dx, dy = wall.coords[-1][0] - wall.coords[0][0], wall.coords[-1][1] - wall.coords[0][1]
        
        # Create normals on both sides of the wall
        normal1 = LineString([midpoint, (midpoint.x - dy, midpoint.y + dx, midpoint.y)])
        normal2 = LineString([midpoint, (midpoint.x + dy, midpoint.y - dx, midpoint.y)])

        # Choose the normal that is outside the polygon
        if not polygon.contains(normal1):
            wall_normal = normal1
        else:
            wall_normal = normal2

        # Calculate angles between the wall and the point and road normal
        wall_point_angle_radians = calculate_angle_between_normal_and_point(wall_normal, receiver_point)
        wall_road_angle_radians = calculate_angle_between_lines(road_normal, wall_normal)
        
        if print_angles:
            print("Angle between wall and point:")
            print(np.degrees(wall_point_angle_radians))
            print(np.around(wall_point_angle_radians,2)/np.pi)
            print("Angle between wall and road:")
            print(np.degrees(wall_road_angle_radians))
            print(np.around(wall_road_angle_radians,2)/np.pi)
            print("Sum of angles:")
            print(np.around((wall_point_angle_radians+wall_road_angle_radians), 2)/np.pi)
            
        if distance_receiver_building > dist_point_road and ((np.around(wall_road_angle_radians,2) <= 0.1 * np.pi) or (np.around(wall_road_angle_radians,2) >= 1.9 * np.pi)) and ((np.around(wall_point_angle_radians,2) >= 0.9 * np.pi) and (np.around(wall_point_angle_radians,2) <= 1.1 * np.pi)):
            not_intersecting, alpha_r_radians, perspective_distance, perspective_center, lines_to_corners = reflection_surface(True, wall, wall_normal, receiver_point,road_segment, buildings_subset)
            if not_intersecting == False or wall.length * np.cos(wall_point_angle_radians) < np.sqrt(2*perspective_distance):
                continue
            return not_intersecting, alpha_r_radians, perspective_center, lines_to_corners, wall_normal
        
        if distance_source_building > dist_point_road:
            if (np.around(wall_road_angle_radians,2) >= 0.9 * np.pi) and (np.around(wall_road_angle_radians,2) <= 1.1 * np.pi):
                not_intersecting, alpha_r_radians, perspective_distance, perspective_center, lines_to_corners = reflection_surface(True, wall, wall_normal, receiver_point,road_segment, buildings_subset)
                if not_intersecting == False or wall.length * np.cos(wall_point_angle_radians) < np.sqrt(2*perspective_distance):
                    continue
                return not_intersecting, alpha_r_radians, perspective_center, lines_to_corners, wall_normal

            if (np.around(wall_road_angle_radians,2) >= 0.4 * np.pi) and (np.around(wall_road_angle_radians,2) <= 1.6 * np.pi):
                not_intersecting, alpha_r_radians, perspective_distance, perspective_center, lines_to_corners = reflection_surface(False, wall, wall_normal, receiver_point,road_segment, buildings_subset)
                if not_intersecting == False or wall.length * np.cos(wall_point_angle_radians) < np.sqrt(2*perspective_distance):
                    continue
                return not_intersecting, alpha_r_radians, perspective_center, lines_to_corners, wall_normal

        if (np.around(wall_road_angle_radians,2) >= 0.40 * np.pi) and (np.around(wall_road_angle_radians,2) <= 1.6 * np.pi) and (np.around((wall_point_angle_radians+wall_road_angle_radians), 2) <= 1.0* np.pi or np.around((wall_point_angle_radians+wall_road_angle_radians), 2) >= 3.0*np.pi):
            not_intersecting, alpha_r_radians, perspective_distance, perspective_center, lines_to_corners = reflection_surface(False, wall, wall_normal, receiver_point,road_segment, buildings_subset)
            if not_intersecting == False or wall.length * np.cos(wall_point_angle_radians) < np.sqrt(2*perspective_distance):
                continue
            return not_intersecting, alpha_r_radians, perspective_center, lines_to_corners, wall_normal
    return False, 0, None, None, None

def check_angles(wall_and_road_facing_ea, facing_wall, angle_normal_point_1, angle_normal_point_2, angle_normal_road_1_1, angle_normal_road_1_2, angle_normal_road_2_1, angle_normal_road_2_2):
        
#     print("angle_normal_road_1_1")
#     print(np.degrees(angle_normal_road_1_1))
#     print("angle_normal_road_1_2")
#     print(np.degrees(angle_normal_road_1_2))
#     print("angle_normal_road_2_1")
#     print(np.degrees(angle_normal_road_2_1))
#     print("angle_normal_road_2_2")
#     print(np.degrees(angle_normal_road_2_2))
#     print("angle_normal_point_1")
#     print(np.degrees(angle_normal_point_1))
#     print("angle_normal_point_2")
#     print(np.degrees(angle_normal_point_2))
    
    

    if (not facing_wall) and (angle_normal_road_1_1 < angle_normal_point_1 or angle_normal_road_2_2 > angle_normal_point_2):
        return False
    
    # Receiver in front of wall, and wall and road are facing each other
    if facing_wall and not wall_and_road_facing_ea:
        if angle_normal_road_1_1 > angle_normal_road_1_2 and angle_normal_road_2_1 > angle_normal_road_2_2:
            if angle_normal_road_2_2 > angle_normal_point_2 or angle_normal_road_1_1 < angle_normal_point_1:
                return False
        else:
            if angle_normal_road_2_1 > angle_normal_point_1 or angle_normal_road_1_2 < angle_normal_point_2:
                return False  
    
    return True

def calculate_angle_between_lines(line1, line2, zero_to_pi=False):
    # Get the x and y components of the vectors representing the lines
    x1, y1 = line1.coords[0][0] - line1.coords[1][0], line1.coords[0][1] - line1.coords[1][1]
    x2, y2 = line2.coords[1][0] - line2.coords[0][0], line2.coords[1][1] - line2.coords[0][1]

    # Calculate the angle between the lines using the atan2 function
    angle = math.atan2(y2, x2) - math.atan2(y1, x1)
    
    p1, p2 = line1.coords[0], line1.coords[1]
    q1, q2 = line2.coords[0], line2.coords[1]

    # calculate the angle between the two lines using the atan2 function
    angle = math.atan2(q2[1]-q1[1], q2[0]-q1[0]) - math.atan2(p2[1]-p1[1], p2[0]-p1[0])

        
    # Ensure that the angle is between 0 and 2*pi
    if angle < 0:
        angle += 2*math.pi
    if zero_to_pi:
        wrapped_angle = angle % (2 * math.pi)
        if wrapped_angle < 0:
            wrapped_angle += 2 * math.pi
        if wrapped_angle > math.pi:
            wrapped_angle = 2 * math.pi - wrapped_angle
        angle = wrapped_angle
    return angle 

def calculate_angle(line):
    # Get the x and y coordinates of the line's endpoints
    x1, y1, _ = line.coords[0]
    x2, y2, _ = line.coords[-1]

    # Calculate the angle using arctangent function
    angle = math.atan2(y2 - y1, x2 - x1)

    # Convert the angle to the range 0 to pi
    return angle



from sklearn.linear_model import LinearRegression
from shapely.geometry import LineString, Point

def determine_reflection_plane(start_point, end_point, terrain_elevations):
    # perform linear regression on z with respect to index of each point       
    
    if len(terrain_elevations) == 1:
        terrain_elevations = [terrain_elevations[0], terrain_elevations[0]]

    
#     distance = np.sqrt((end_point[0]-start_point[0])**2 + (end_point[1]-start_point[1])**2)
#     distances = np.arange(0, distance, 2)
#     print(distances)
#     print(np.arange(len(terrain_elevations[1:]))*2)
# #     print(len(terrain_elevations))
    slope, intercept = np.polyfit(np.arange(len(terrain_elevations)), terrain_elevations, 1)



#     if len(terrain_elevations) == 1:
#         terrain_elevations = [terrain_elevations[0], terrain_elevations[0]]
#     regressor = LinearRegression().fit(np.arange(len(terrain_elevations[1:])).reshape(-1, 1), terrain_elevations[1:])
#     slope = regressor.coef_[0]
#     intercept = regressor.intercept_
    

    # calculate z values for endpoints of line
    start_z = slope * 0 + intercept
    end_z = slope * (len(terrain_elevations[1:]) - 1) + intercept

    # create LineString object with new endpoints
    new_line = LineString([(start_point[0], start_point[1], start_z), (end_point[0], end_point[1], end_z)])
    return new_line, slope, intercept

# Function that gets the indices of the largest values of an array. Useful when dealing with buildings

def get_max_indices(array, threshold=15):
    max_val = np.max(array)
    #     filtered = np.where(np.logical_and(array >= max_val - threshold, array <= max_val + threshold))[0]
    filtered = np.where(np.logical_and(array + threshold >= max_val, array>=0.5))[0]
#     print(test)
#     filtered = np.where(array >= 1)[0]
#     print(filtered)
    if filtered.size == 0:
        max_idx = np.argmax(array)
        return np.array([max_idx, max_idx])
    else:
        return np.array([filtered[0], filtered[-1]])

def get_subset_of_buildings(buildings_gdf, connecting_line, buffer_size):
    # create a buffer around the connecting line
    buffer = connecting_line.buffer(buffer_size)

    # Get the subset of buildings that are within the buffer
    subset = buildings_gdf[buildings_gdf.geometry.intersects(buffer)]
    
    return subset

def calculate_normal(line: LineString, point: Point, opposite: bool = False) -> LineString:
    # Calculate the normal of a line pointing towards or away from a line
    midpoint = line.interpolate(0.5, normalized=True)
    center_to_point = np.array([point.x - midpoint.x, point.y - midpoint.y])
    dx, dy = line.coords[-1][0] - line.coords[0][0], line.coords[-1][1] - line.coords[0][1]
    normal = np.array([-dy, dx])
    
    if np.dot(normal, center_to_point) < 0:
        if opposite:
            return LineString([midpoint.coords[0][0:2], (midpoint.x + normal[0], midpoint.y + normal[1])])
        return LineString([midpoint.coords[0][0:2], (midpoint.x - normal[0], midpoint.y - normal[1])])
    if opposite:
        return LineString([midpoint.coords[0][0:2], (midpoint.x - normal[0], midpoint.y - normal[1])])
    return LineString([midpoint.coords[0][0:2], (midpoint.x + normal[0], midpoint.y + normal[1])])


def calculate_angle_between_normal_and_point(normal_line, point):
    point_to_normal = LineString([normal_line.coords[0][0:2], point.coords[0][0:2]])
    return calculate_angle_between_lines(point_to_normal, normal_line)