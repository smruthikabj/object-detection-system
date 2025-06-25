import os
from flask import Flask, request, send_file, render_template
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import label_map_util # Utility to visualize detection
from object_detection.utils import visualization_utils as viz_utils # Utility to visualize detection

app = Flask(__name__)

# Load your object detection model
model_path = os.path.abspath(os.path.join('ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'saved_model'))
print(f"Model path: {model_path}")
model = tf.saved_model.load(model_path)

# Load the label map
label_map_path = 'mscoco_label_map.pbtxt'  # Update this path if necessary
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = Image.open(file.stream)

    # Convert image to numpy array for processing
    image_np = np.array(image)

    # Add a batch dimension since the model expects a batch of images
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run object detection
    detections = model(input_tensor)

    # Extract detection details
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # Only keep detections with scores above a threshold
    threshold = 0.5
    for i in range(detection_boxes.shape[0]):
        if detection_scores[i] >= threshold:
            box = detection_boxes[i]
            y1, x1, y2, x2 = box
            cv2.rectangle(image_np, (int(x1 * image_np.shape[1]), int(y1 * image_np.shape[0])),
                          (int(x2 * image_np.shape[1]), int(y2 * image_np.shape[0])), (0, 255, 0), 2)
            
            # Get class name
            class_name = category_index[detection_classes[i]]['name']
            label = f"{class_name}: {detection_scores[i]:.2f}"

            # Annotate the image with class name
            cv2.putText(image_np, label, (int(x1 * image_np.shape[1]), int(y1 * image_np.shape[0]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert back to PIL image
    result_image = Image.fromarray(image_np)

    # Save to a BytesIO object and send back to client
    img_io = io.BytesIO()
    result_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)