import requests
import json
import io
import geopandas as gpd
import datetime as dt
import numpy as np
import pandas as pd


def compose_date(years, months=1, days=1, weeks=None, hours=None, minutes=None,
                 seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)


def read_csv_for_traffic(url):
    try:
        #dtypes = {"tmsNumber": int, "year": int, "doy": int, "h": int, "m": int, "s": int, "ms": int, "pituus (m)": float, "kaista": int, "suunta": int, "class": int, "nopeus (km/h)": float, "faulty": int, "kokonaisaika (tekninen)": int, "aikaväli (tekninen)": int, "jonoalku (tekninen)": int}
        traffic_cols = ["tmsnumber","year","doy","h","m","s","ms","pituus (m)","kaista","suunta","class","nopeus (km/h)","faulty","kokonaisaika (tekninen)","aikaväli (tekninen)","jonoalku (tekninen)"]
        return pd.read_csv(url, names=traffic_cols, sep=";")
    except:
        return None


def create_traffic_df_by_hour(traffic_stations, first_day=1, last_day=dt.datetime.now().timetuple().tm_yday):
    tmsnumbers = traffic_stations['tmsNumber'].values.tolist()

    traffic_by_day = []
    
    for i in range(first_day, last_day):
        urls = (f"https://tie.digitraffic.fi/api/tms/v1/history/raw/lamraw_{tmsnumber}_23_{i}.csv" for tmsnumber in tmsnumbers)
        traffic_df = pd.concat(map(read_csv_for_traffic, urls))
        if traffic_df is None:
            continue
        traffic_df['year'] = traffic_df['year'] + 2000
        traffic_df['date'] = compose_date(traffic_df['year'], days=traffic_df['doy'], hours=traffic_df['h'], minutes=traffic_df['m'], seconds=traffic_df['s'], milliseconds=traffic_df['ms'])
        traffic_df = traffic_df[traffic_df['faulty']==0]
        
        result_df = traffic_df.groupby(['h'])['nopeus (km/h)'].mean()
        result_df = traffic_df.groupby([pd.Grouper(key='date',freq='H'),traffic_df['class'], traffic_df['tmsnumber']]).mean()['nopeus (km/h)']
        result_df = result_df.reset_index()

        by_hour = traffic_df.groupby([pd.Grouper(key='date',freq='H'),traffic_df['class'], traffic_df['tmsnumber']]).size().reset_index(name='count')
        by_hour['speed'] = result_df['nopeus (km/h)']
        traffic_by_day.append(by_hour)
    
    return pd.concat(traffic_by_day, ignore_index=True)

stations = requests.get("https://tie.digitraffic.fi//api/tms/v1/stations")
stations_t = json.load(io.StringIO(stations.text))
weather = requests.get("https://tie.digitraffic.fi/api/weather/v1/stations")
weather_t = json.load(io.StringIO(weather.text))
gdf_weather = gpd.GeoDataFrame.from_features(weather_t["features"])
gdf = gpd.GeoDataFrame.from_features(stations_t["features"])
#ax = gdf_weather.plot()
gdf.crs = "EPSG:4326"
tampere = pd.concat([gdf.loc[["tampere" in c.lower() for c in  list(gdf['name'])]], gdf.loc[["tre" in c.lower() for c in  list(gdf['name'])]], gdf.loc[["rautaharkko" in c.lower() for c in  list(gdf['name'])]]])
# Not in Tampere
tampere = tampere[tampere.name != "vt7_Treksilä"] 
tampere = tampere[tampere.name != "vt3_Tampere_Myllypuro"]

tampere = tampere.to_crs(4326)

traffic_df_hour = create_traffic_df_by_hour(tampere)
traffic_df_hour = traffic_df_hour[traffic_df_hour['class']!=0]
# Pivot the table to get the desired format
pivot = pd.pivot_table(traffic_df_hour, values=['count', 'speed'], index=['date', 'tmsNumber'], columns='class')

# Flatten the MultiIndex columns
pivot.columns = [f"{col[0]}_{col[1]}" if len(col) == 2 else f"{col[0]}_{col[1]}_{col[2]}" for col in pivot.columns]

# Reset the index
pivot = pivot.reset_index()
pivot = pivot.fillna(0)

pivot['count_light'] = np.sum(pivot[['count_1', 'count_6', 'count_7']].values, axis=1)
pivot['count_heavy'] = np.sum(pivot[['count_2', 'count_3', 'count_4', 'count_5', 'count_9']].values, axis=1)
pivot['speed_light'] = (pivot['count_1']*pivot['speed_1'] + pivot['count_6']*pivot['speed_6'] + pivot['count_7']*+pivot['speed_7']) / pivot['count_light']
pivot['speed_heavy'] = (pivot['count_2']*pivot['speed_2'] + pivot['count_3']*pivot['speed_3'] + pivot['count_4']*pivot['speed_4'] + pivot['count_5']*pivot['speed_5'] + pivot['count_9']*pivot['speed_9']) / pivot['count_heavy']

traffic = pivot.copy()
# traffic_df_hour.to_csv('traffic_per_hour.csv')  
traffic.to_csv('traffic.csv')