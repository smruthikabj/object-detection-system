import os
from flask import Flask, request, send_file, render_template, redirect, url_for
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

app = Flask(__name__)

# Load your object detection model
model_path = os.path.abspath(os.path.join('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'saved_model'))
print(f"Model path: {model_path}")
model = tf.saved_model.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = Image.open(file.stream)

    # Convert image to numpy array for processing
    image_np = np.array(image)

    # Add a batch dimension
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run object detection
    detections = model(input_tensor)

    # Extracting detection results for rendering
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    # Prepare data for the results table
    detection_data = []
    for i in range(len(detection_classes)):
        detection_data.append({
            'class': detection_classes[i],
            'score': detection_scores[i],
            'box': detection_boxes[i].tolist()
        })

    # Visualization of the results of a detection
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Convert back to PIL image
    result_image = Image.fromarray(image_np)

    # Save to a BytesIO object
    img_io = io.BytesIO()
    result_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return render_template('results.html', image_data=img_io.getvalue(), detections=detection_data)

if __name__ == '__main__':
    app.run(debug=True)