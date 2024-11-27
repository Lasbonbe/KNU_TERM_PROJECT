import tensorflow as tf

# ⚠️ THIS CODE DOES NOT WORK WITH TENSORFLOW 2.0 OR ABOVE ⚠️
# ⚠️ USE TENSORFLOW 1.15.0 TO RUN THIS CODE ⚠️
# This code snippet is used to convert a TensorFlow model to a .pb file
# This is useful for converting a model to a format that can be used in OpenCV
# Replace the saved_model_dir and output_node_names with your model's path and output nodes

saved_model_dir = "models/datasetV2"
output_node_names = ["output_node_name"]  # Replace with your model's output nodes
frozen_graph_path = "frozen_model.pb"

# Freeze the model
converter = tf.compat.v1.graph_util.convert_variables_to_constants
frozen_func = converter(
    tf.compat.v1.Session(),
    tf.compat.v1.get_default_graph().as_graph_def(),
    output_node_names,
)

# Save to .pb file
with tf.io.gfile.GFile(frozen_graph_path, "wb") as f:
    f.write(frozen_func.SerializeToString())