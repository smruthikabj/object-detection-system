import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils as viz_utils

# Load your object detection model
model_path = os.path.abspath('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
model = tf.saved_model.load(model_path)

# Load the label map
label_map_path = os.path.abspath('mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Function to detect objects in a frame
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = model(input_tensor)

    # Process detection results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Detection classes should be integers
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    detections = detect_objects(rgb_frame)

    # Visualize detection results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Display the frame with detections
    cv2.imshow('Object Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

