# py-noise-prediction

This code can be used for calculating noise levels near major roadways in Tampere. 
The code is not very well organized and may require some sorting to make the running of the code more straightforward.
With some work, it should be possible to calculate noise levels in any area of finland with traffic monitoring stations. https://www.digitraffic.fi/en/road-traffic/lam/


Calculating noise levels requires an elevation raster and traffic data. The code for downloading them is included in the load_and_combine_raster.py and traffic.py -files.
The create bbox function in the load_and_combine_raster.py can be used to create bounding boxes of the appropriate size for different areas in Finland.


THe main code is in process_gis_data, which uses the different functions to retrieves the roads on which the traffic monitoring stations are on, and cuts them up into sections.
Then, calculation points are spread around the roads, and noise deltas are calculated between the roads and the calculation points. 
Finally, an SQL database connection is made and the relevant data is transferred into the database.
The code for processing the delta data, as well as the traffic data, is included in the SQL files.
First, the L1 values should be calculated, then the hourly values for each day, and finally the days are averaged to get the entire week's continuous average.



# The code was originally written as a jupyter notebook. It might be useful to split the code into more separate files
