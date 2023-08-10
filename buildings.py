import itertools
import requests
import json
import io
import shapely
import geopandas as gpd
import pandas as pd
import numpy as np
from processing_funs import get_elevation_at_lat_lon


def add_z_to_buildings(row):#.coords.xy
    xs, ys = row['geometry'].exterior.coords.xy
    points = []
    for x, y in zip(xs, ys):
        points.append([x, y, row['pohjankorkeus']/1000])
    #print(points)
    return shapely.geometry.Polygon(points)

def add_z_to_tam_buildings(row):#.coords.xy
    xs, ys = row['geometry'].exterior.coords.xy
    cent_x, cent_y = row['geometry'].centroid.coords[0]
    elevation = get_elevation_at_lat_lon('mosaic.tif', cent_x, cent_y)
    points = []
    for x, y in zip(xs, ys):
        points.append([x, y, elevation])
    #print(points)
    return shapely.geometry.Polygon(points)

def get_building_data_country(roads, api_key):
    url = f"https://avoin-paikkatieto.maanmittauslaitos.fi/maastotiedot/features/v1/collections/rakennus/items?&api-key={api_key}&bbox=321194,6814792,339103,6827099&bbox-crs=http://www.opengis.net/def/crs/EPSG/0/3067&crs=http://www.opengis.net/def/crs/EPSG/0/3067"

    # The data is split into multiple pages
    # Go through the page, then continue to the next until the end is reached
    iterator = 0
    while True:
        mml_buildings_resp = requests.get(url)
        buildings_json = json.load(io.StringIO(mml_buildings_resp.text))
        for link in buildings_json["links"]:
            #print(link)
            if link['rel'] == 'next':
                #print('changing')
                url_next = link['href']
        
        sub_buildings_gdf = gpd.GeoDataFrame.from_features(buildings_json["features"])
        if iterator > 0:
            buildings_gdf = pd.concat([buildings_gdf, sub_buildings_gdf])
        else:
            buildings_gdf = sub_buildings_gdf
        if url_next == url:
            break
        url = url_next
        iterator += 1

    buildings_gdf.crs = "EPSG:3067"

    buildings_gdf['geometry'] = buildings_gdf.apply(add_z_to_buildings, axis=1)
    buildings_gdf['roof_elevation'] = (buildings_gdf['pohjankorkeus']/1000) + 10 * buildings_gdf['kerrosluku']

    road_buffer = roads.buffer(250)

    keep_rows = []
    for i, row in buildings_gdf.iterrows():
        for geom in road_buffer:
            if row.geometry.within(geom):
                keep_rows.append(row)
                break

    buildings = gpd.GeoDataFrame(data=keep_rows, geometry='geometry', crs="epsg:3067")
    return buildings


def add_roof_elevation_tam(row):
    # These are sheds, garages etc.
    if np.isnan(row['F_KERROSTEN_LKM']):
        return row['geometry'].exterior.coords[0][2] + 3
    # Most of these are either single family homes with thatched roofs or industrial buildings etc.
    elif row['F_KERROSTEN_LKM'] == 1:
        return row['geometry'].exterior.coords[0][2] + 8
    # For high rises, 1 + 3 * floors. Take into account the roof and ground floor
    else:
        return 2 + row['geometry'].exterior.coords[0][2] + 3 * row['F_KERROSTEN_LKM']


def get_building_data_tampere(roads):
    # Tampere specific building data
    url = f"https://geodata.tampere.fi/geoserver/rakennukset/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=rakennukset:RAKENN_ST_FA_GSVIEW&outputFormat=json"

    tam_buildings_resp = requests.get(url)
    tam_buildings_json = json.load(io.StringIO(tam_buildings_resp.text))
    tam_buildings_gdf = gpd.GeoDataFrame.from_features(tam_buildings_json["features"])

    tam_buildings_gdf.crs = "EPSG:3878"
    tam_buildings_gdf = tam_buildings_gdf.to_crs("EPSG:3067")

    # Take a subset of buildings 250 meters around the studied roads
    road_buffer = roads.buffer(250)
    keep_rows = []
    for i, row in tam_buildings_gdf.iterrows():
        for geom in road_buffer:
            if row.geometry.intersects(geom):
                keep_rows.append(row)
                break

    tam_buildings_filt = gpd.GeoDataFrame(data=keep_rows, geometry='geometry', crs="epsg:3067")
    tam_buildings_filt = tam_buildings_filt.explode()

    # Remove possible duplicate buildings
    for a, b in itertools.combinations(tam_buildings_filt.geometry, 2):
        if a.equals(b) == True:
            try:
                tam_buildings_filt.drop(tam_buildings_filt[tam_buildings_filt['geometry'] == b].index.values[1:], inplace=True)
            except:
                continue
    forbidden_use_cases = ['163', '941', 'E01']  # this should remove underground structures
    tam_buildings_filt = tam_buildings_filt[~(tam_buildings_filt['F_KAYTTOTARK'].isin(forbidden_use_cases))]

    tam_buildings_filt['geometry'] = tam_buildings_filt.apply(add_z_to_tam_buildings, axis=1)
    tam_buildings_filt['roof_elevation'] = tam_buildings_filt.apply(add_roof_elevation_tam, axis=1)

    return tam_buildings_filt.copy()
