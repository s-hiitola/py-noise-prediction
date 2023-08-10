# The National Land Survey has a limit on the size of raster data from a single request
# This helper code downloads multiple height data rasters, and combines them into one

import rasterio
import requests


API_KEY = "d073a2c2-76f0-48bf-9279-847e72168c7f"

bbox = {
    "nw" : [321194,6820945,330149,6827099],
    "sw" : [321194,6814792,330149,6820945],
    "ne" : [330149,6820945,339103,6827099],
    "se" : [330149,6814792,339103,6820945]
}

heightmaps = {}

for box, coords in bbox.items():
    x1,y1,x2,y2 = coords
    url = f"https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2?service=WCS&version=2.0.1&request=GetCoverage&api-key={API_KEY}&CoverageID=korkeusmalli_2m&SUBSET=E({x1},{x2})&SUBSET=N({y1},{y2})&format=image/tiff&geotiff:compression=LZW"
    #print(box, x1,x2,y1,y2)
    # print(url)
    r = requests.get(url, stream = True)

    with open(f"{box}.tiff","wb") as geotiff:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                geotiff.write(chunk)
                    
# Define the paths to the four geotiffs
paths = ["nw.tiff", "ne.tiff", "sw.tiff", "se.tiff"]

# Open each geotiff and read the metadata
src_files_to_mosaic = []
for path in paths:
    src = rasterio.open(path)
    src_files_to_mosaic.append(src)

# Merge the four geotiffs into a single raster
mosaic, out_trans = rasterio.merge.merge(src_files_to_mosaic)

# Update the metadata for the mosaic
out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": src.crs})

# Write the mosaic to a new geotiff file
with rasterio.open("mosaic.tif", "w", **out_meta) as dest:
    dest.write(mosaic)
