import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util, visualization_utils as viz_utils

# Load the model
model_path = 'path_to_your_model_directory/saved_model'
detect_fn = tf.saved_model.load(model_path)

# Load label map
label_map_path = 'path_to_your_label_map/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

image_path = 'path_to_your_image.jpg'
image_np = load_image_into_numpy_array(image_path)

# Perform detection
input_tensor = tf.convert_to_tensor(image_np)
input_tensor = input_tensor[tf.newaxis, ...]

detections = detect_fn(input_tensor)

# Extract detection data
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections

# Detection classes should be integers
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

# Get detected items and their confidence scores
detected_items = []
for i in range(num_detections):
    class_id = detections['detection_classes'][i]
    score = detections['detection_scores'][i]
    if score >= 0.5:  # Confidence threshold
        item = category_index[class_id]['name']
        detected_items.append((item, score))

# Print detected items in a table
print(f"{'Item':<15}{'Confidence':<10}")
print("-" * 25)
for item, score in detected_items:
    print(f"{item:<15}{score:.2f}")

