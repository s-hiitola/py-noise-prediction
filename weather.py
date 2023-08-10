import datetime as dt
import pandas as pd
from fmiopendata import download_stored_query

# Set the time period to retrieve the data
start_time = dt.datetime(year=2023, month=1, day=1)
end_time = dt.datetime.utcnow()

# Split the time period into smaller periods of 160 hours or less
periods = []
delta = dt.timedelta(hours=160)
while start_time < end_time:
    periods.append((start_time, min(start_time + delta, end_time)))
    start_time += delta

# Retrieve the data for each period and concatenate the resulting DataFrames
dfs = []
for period in periods:
    obs = download_stored_query("fmi::observations::weather::multipointcoverage",
                                args=["place=Tampere",
                                      "starttime=" + period[0].isoformat(timespec="seconds") + "Z",
                                      "endtime=" + period[1].isoformat(timespec="seconds") + "Z",
                                      "timeseries=True"])
    rearranged = {}
    for key, value in obs.data['Tampere Siilinkari'].items():
        if type(value) == dict:
            key = f"{key} {value['unit']}"
            value = (value['values'])
        rearranged[key] = value
    df = pd.DataFrame(rearranged)
    dfs.append(df)

weather = pd.concat(dfs, ignore_index=True)
weather.head()

hourly_weather = weather.resample('H', on='times').mean()
