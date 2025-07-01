# import tensorflow as tf
# import coremltools as ct

# # 1. Load the HDF5 checkpoint
# model = tf.keras.models.load_model(
#     "/Users/sebastianscrimenti/Documents/Models/imageseg_canopy_model/imageseg_canopy_model.hdf5",
#     compile=False)

# # 2. Convert directly
# mlmodel = ct.convert(
#     model,
#     source="tensorflow",           # safe even when autodetect works
#     inputs=[ct.ImageType(
#         name="image",
#         shape=(1, 256, 256, 3),
#         scale=1/255.0,
#         bias=0.0)],
#     minimum_deployment_target=ct.target.iOS14)

# # 3. Save
# mlmodel.save("CanopyModel.mlmodel")
# print("✓  Core ML model written to CanopyModel.mlmodel")


# import os

# # os.environ["TF_USE_LEGACY_KERAS"]="1"

# import tensorflow as tf
# h5_path = "/Users/sebastianscrimenti/Documents/Models/imageseg_canopy_model/imageseg_canopy_model.hdf5"           # your file
# model = tf.keras.models.load_model(h5_path, compile=False)

# print(model.input_shape)   # e.g. (None, 256, 256, 3)
# print(model.output_shape)  # e.g. (None, 256, 256, 1)

# import coremltools as ct

# H, W = 256, 256   # replace with your training resolution

# image_in = ct.ImageType(
#         name="image",
#         shape=(1, H, W, 3),        # NHWC
#         scale=1/255.0,             # same normalisation you trained with
#         bias=[0.0, 0.0, 0.0])

# # Single-channel mask → grayscale image output
# mask_out = ct.ImageType(
#         name="mask",
#         shape=(1, H, W, 1),
#         color_layout=ct.colorlayout.GRAYSCALE)

# mlmodel = ct.convert(
#         model,
#         convert_to="mlprogram",      # iOS 14+ / macOS 11+
#         source="tensorflow",          # safe even when autodetect works
#         inputs=[image_in],
#         # outputs=[mask_out],
#         compute_units=ct.ComputeUnit.ALL)

# mlmodel.save("Segmentation.mlpackage")   # Xcode 15+ prefers .mlpackage


# import coremltools as ct 
# import tensorflow as tf

# h5_path = "/Users/sebastianscrimenti/Documents/Models/imageseg_canopy_model/imageseg_canopy_model.hdf5"           # your file
# tf_model = tf.keras.models.load_model(h5_path, compile=False)


# print(tf_model.input_shape)   # e.g. (None, 256, 256, 3)
# print(tf_model.output_shape)  # e.g. (None, 256, 256, 1)

# model = ct.convert(tf_model, source = "tensorflow")
# model.save("Segmentation.mlpackage")   # Xcode 15+ prefers .mlpackage

# import numpy as np
# x = np.random.rand(1, 256, 256, 3)
# tf_out = model.predict([x])

import coremltools as ct
import tensorflow as tf
import numpy as np

# 1. Load your TensorFlow model
h5_path = "/Users/sebastianscrimenti/Documents/Models/imageseg_canopy_model/imageseg_canopy_model.hdf5"
tf_model = tf.keras.models.load_model(h5_path, compile=False)

print(f"--- Actual TF output tensor name: {tf_model.output.name} ---")

# 2. Define the model's INPUT
image_input = ct.ImageType(
    name="input_img",
    shape=(1, 256, 256, 3),
    scale=1/255.0,
    color_layout="RGB"
)

# 3. Convert the model, letting Core ML Tools infer the output
coreml_model = ct.convert(
    tf_model,
    inputs=[image_input],
    source="tensorflow",
    minimum_deployment_target=ct.target.iOS15
)

# 4. Save the successfully converted Core ML model
output_path = "CanopySegmentation.mlpackage"
coreml_model.save(output_path)
print(f"\n✓ Core ML model saved to {output_path}")

# 5. Inspect the final model to see what the output was named
print("\n--- Converted Core ML Model Spec ---")
print(coreml_model)

# 6. Test the model using the CORRECT method to get the output name
print("\n--- Testing the converted model ---")
output_name = coreml_model.outputs[0].name
print(f"Core ML model created with output name: '{output_name}'")

test_image = np.random.rand(1, 256, 256, 3).astype(np.float32)
input_data = {"input_img": test_image}
coreml_output = coreml_model.predict(input_data)

output_mask = coreml_output[output_name]
print(f"Shape of the output mask: {output_mask.shape}")
print("✓ Conversion and prediction test successful!")