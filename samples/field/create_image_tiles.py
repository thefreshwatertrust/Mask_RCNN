import fiona
import rasterio as rio
from rasterio.mask import mask
from rasterio.plot import show
from shapely import geometry
from glob import glob
from numpy.random import rand

# Data folder: https://thefreshwatertrustorg-my.sharepoint.com/personal/dion_thefreshwatertrust_org/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdion%5Fthefreshwatertrust%5Forg%2FDocuments%2FDocuments%2FGrandview%5FOrtho%5F2019&originalPath=aHR0cHM6Ly90aGVmcmVzaHdhdGVydHJ1c3RvcmctbXkuc2hhcmVwb2ludC5jb20vOmY6L2cvcGVyc29uYWwvZGlvbl90aGVmcmVzaHdhdGVydHJ1c3Rfb3JnL0VtOGhURS1zVkJOQ2lUTjdvVTBxN1JZQmJ3VjdISGhsSllrcWhzblJpaldPdnc%5FcnRpbWU9aEp1Smp2QTcyVWc

# Collect shape geometries from shapefile
sf = fiona.open("Grandview_fields/Grandview_fields_DW.shp", "r")

# Open raster image file
#rf = rasterio.open("/Users/dharp/Desktop/gv_1500dpi.tif")
rf = rio.open("Grandview_1500dpi.tif")

# Set buffer
N = 2048 # pixels

for s in sf:
    if s["geometry"]["type"] == "MultiPolygon": continue
    if len(s["geometry"]["coordinates"]) != 1: continue

    # Split into training and validation sets
    if rand() <= 0.8:
        outfile = r'train/gvclip_shapeid{}.tif'
    else:
        outfile = r'val/gvclip_shapeid{}.tif'
    
    # Get pixel ids at center of shape
    pshape = geometry.Polygon(s["geometry"]["coordinates"][0])
    xc = pshape.centroid.coords.xy[0][0]
    yc = pshape.centroid.coords.xy[1][0]
    py, px = rf.index(xc, yc)

    # Create clipping window
    window = rio.windows.Window(px - N//2, py - N//2, N, N)

    # Clip raster and set metadata
    clip = rf.read(window=window)
    out_meta = rf.meta.copy()
    out_meta.update({"height": N,
                     "width": N,
                     "transform": rio.windows.transform(window, rf.transform)})

    # Save clip
    with rio.open(outfile.format(s['id']), "w", **out_meta) as dest: dest.write(clip)

