import coremltools as ct
import tensorflow as tf

# 1) Load your HDF5 model for inference only
keras_model = tf.keras.models.load_model(
    "imageseg_canopy_model.hdf5",
    compile=False
)

# 2) Define the Core ML input spec
#    – make sure the name here exactly matches your model’s input tensor (e.g. “input_img”)
#    – swap out 256×256×3 if your model uses a different shape
input_spec = ct.TensorType(
    shape=(1, 256, 256, 3),
    name="input_img"
)

# 3) Convert to the modern MLProgram format
mlmodel = ct.convert(
    keras_model,
    inputs=[input_spec],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)

# 4) Save your .mlmodel package
mlmodel.save("imageseg_canopy_model.mlpackage")
print("✅  Saved imageseg_canopy_model.mlpackage")
