import tf2onnx
import tensorflow as tf

#********************************************************************************************************************
#░░░██╗░░██╗███████╗        ████████╗░█████╗░       ░░░░█████╗░███╗░░██╗███╗░░██╗██╗░░██╗
#░░░██║░░██║██╔════╝        ╚══██╔══╝██╔══██╗       ░░░██╔══██╗████╗░██║████╗░██║╚██╗██╔╝
#░░░███████║██████╗░        ░░░██║░░░██║░░██║       ░░░██║░░██║██╔██╗██║██╔██╗██║░╚███╔╝░
#░░░██╔══██║╚════██╗        ░░░██║░░░██║░░██║       ░░░██║░░██║██║╚████║██║╚████║░██╔██╗░
#██╗██║░░██║██████╔╝        ░░░██║░░░╚█████╔╝       ██╗╚█████╔╝██║░╚███║██║░╚███║██╔╝╚██╗
#╚═╝╚═╝░░╚═╝╚═════╝░        ░░░╚═╝░░░░╚════╝░       ╚═╝░╚════╝░╚═╝░░╚══╝╚═╝░░╚══╝╚═╝░░╚═╝
# This code snippet is used to convert a TensorFlow model to an ONNX model
# Because OpenCV does not support  .h5 files, we need to convert the model to an ONNX format
# Replace the MODELNAME with the name of your model
#********************************************************************************************************************

MODELNAME = "datasetV2"

# MODEL .h5
model = tf.keras.models.load_model(f"models/{MODELNAME}.h5")


onnx_model = tf2onnx.convert.from_keras(model)

# MODEL ONNX
with open(f"ONNX_models/{MODELNAME}.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())