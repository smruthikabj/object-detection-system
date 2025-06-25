import os
import tensorflow as tf

model_directory = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
model_path = os.path.abspath(model_directory)
print(f"Model path: {model_path}")

if not os.path.exists(model_path):
    print(f"Directory does not exist: {model_path}")

print("Directory contents:", os.listdir(model_path))

saved_model_file = os.path.join(model_path, 'saved_model.pb')
if not os.path.isfile(saved_model_file):
    print(f"File 'saved_model.pb' does not exist in the directory: {saved_model_file}")
else:
    print(f"File 'saved_model.pb' found at: {saved_model_file}")

model = tf.saved_model.load(model_path)
print("Model loaded successfully.")