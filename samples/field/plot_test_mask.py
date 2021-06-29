from field import FieldDataset
import numpy as np
from mrcnn import visualize

# Create FieldDataset Object
dataset_train = FieldDataset()
# Read in the training data
dataset_train.load_fields('.','train')
# Prepare data (load_mask, etc.)
dataset_train.prepare()

# Get an image_id (there is only one currently)
image_id = dataset_train.image_ids[0]

# Retrieve image
image = dataset_train.load_image(image_id)
# Retrieve masks and irrigation types (class_ids)
mask, class_ids = dataset_train.load_mask(image_id)
# Plot
visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
