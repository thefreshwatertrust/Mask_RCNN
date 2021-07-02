"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import fiona
from glob import glob
from shapely import geometry
import rasterio.features
from rasterio.mask import mask

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class FieldConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "field"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 4  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MAX_DIM = 2048

    IMAGE_MIN_DIM = 2048

    IMAGE_SHAPE = [2048, 2048, 3]


############################################################
#  Dataset
############################################################

class FieldDataset(utils.Dataset):

    def load_fields(self, dataset_dir, subset, shapefile):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("field", 1, "Not Irrigated")
        self.add_class("field", 2, "Flood")
        self.add_class("field", 3, "Sprinkler")

        # Train or validation dataset?
        assert subset in ["train", "val"]

        dataset_dir = os.path.join(dataset_dir, subset)

        # Open shapefile containing field polygons
        sf = fiona.open(shapefile, "r")

        # Loop through geotiffs in folder
        for tf in glob(os.path.join(dataset_dir,"*.tif")):
            # Create empty polygons list
            polygons = []
            polygon_types = []

            # Open raster image
            rf = rasterio.open(tf)

            # Get bounds of raster
            bnds = get_raster_bounds(rf)["coordinates"][0]

            # Create shapely polygon from raster bounds
            prf = geometry.Polygon(bnds)

            # Loop through field polygons in shapefile
            for p in sf:
                # Skip polygons with more than 1 set of coordinates
                if len(p['geometry']['coordinates']) != 1: continue
                # Ignore 'MultiPolygon' type for now
                if p['geometry']['type'] == 'Polygon':
                    psf = geometry.Polygon(p['geometry']['coordinates'][0])
                    if prf.contains(psf): 
                        polygons.append(p)

            self.add_image(
                "field",
                image_id=tf, # Just use name of tif for id 
                path=tf,
                width=rf.width, height=rf.height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "field":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        out_mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            psf = geometry.Polygon(p['geometry']['coordinates'][0])
            rf = rasterio.open(info["path"])
            out_image, out_transform = mask(rf, [psf], invert=True)
            out_mask[:,:,i] = out_image[3,:,:]
            rf.close()
            if p['properties']['_irrType'] is None:
                irrType = "Not Irrigated"
            else:
                irrType = p['properties']['_irrType']
            class_ids.append(self.class_names.index(irrType))

        out_mask[out_mask==0] = 1
        out_mask[out_mask==255] = 0

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return out_mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "field":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def get_raster_bounds(rf):
    # Read the dataset's valid data mask as a ndarray.
    mask = rf.dataset_mask()

    # Extract feature shapes and values from the array.
    for geom, val in rasterio.features.shapes(mask, transform=rf.transform):
        # Transform shapes from the dataset's own coordinate
        # reference system to CRS84 (EPSG:4326).
        #geom = rasterio.warp.transform_geom(rf.crs, 'EPSG:4326', geom, precision=6)
        geom = rasterio.warp.transform_geom(rf.crs, rf.crs, geom, precision=6)
        return geom


#def train(model):
#    """Train the model."""
#    # Training dataset.
#    dataset_train = BalloonDataset()
#    dataset_train.load_balloon(args.dataset, "train")
#    dataset_train.prepare()
#
#    # Validation dataset
#    dataset_val = BalloonDataset()
#    dataset_val.load_balloon(args.dataset, "val")
#    dataset_val.prepare()
#
#    # *** This training schedule is an example. Update to your needs ***
#    # Since we're using a very small dataset, and starting from
#    # COCO trained weights, we don't need to train too long. Also,
#    # no need to train all layers, just the heads should do it.
#    print("Training network heads")
#    model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE,
#                epochs=30,
#                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':

    COCO_MODEL_PATH = os.path.join("../../mask_rcnn_coco.h5")

    # Create training dataset
    # Create FieldDataset Object
    dataset_train = FieldDataset()
    # Read in the training data
    dataset_train.load_fields('/Users/dharp/Data/GrandView','train', "/Users/dharp/Data/GrandView/Grandview_fields/Grandview_fields_DW.shp")
    # Prepare data (load_mask, etc.)
    dataset_train.prepare()
    
    # Create validation dataset
    # Create FieldDataset Object
    dataset_val = FieldDataset()
    # Read in the training data
    dataset_val.load_fields('/Users/dharp/Data/GrandView','val', "/Users/dharp/Data/GrandView/Grandview_fields/Grandview_fields_DW.shp")
    # Prepare data (load_mask, etc.)
    dataset_val.prepare()

    config = FieldConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir='model')

    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='all')


