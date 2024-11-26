import tf2onnx
import tensorflow as tf

MODELNAME = "datasetV2"

# MODEL .h5
model = tf.keras.models.load_model(f"models/{MODELNAME}.h5")


onnx_model = tf2onnx.convert.from_keras(model)

# MODEL ONNX
with open(f"ONNX_models/{MODELNAME}.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())