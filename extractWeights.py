import os
import numpy as np
import tensorflow as tf
from tfModel import ZF_UNET_224
# Load the model
model = ZF_UNET_224()
h5_model_path = "zf_unet_224.h5"
model.load_weights(h5_model_path)

weights_dir = "pretrainedKernels_"
os.makedirs(weights_dir, exist_ok=True)

def model_summary_to_markdown(model, filename='model_summary.md'):
    """Saves Keras model summary to a markdown file."""
    with open(filename, 'w') as f:
        f.write('## Model Summary\n\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))

model_summary_to_markdown(model)

for layer in model.layers:
    weights = layer.get_weights()

    if weights:
        print(f"Layer: {layer.name}, Shape(s): {[w.shape for w in weights]}")
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights, bias = layer.get_weights()  # Extract kernel and bias
            # Reshape kernel weights from (H, W, In_C, Out_C) → (Out_C, In_C, H, W)
            reshaped_weights = np.transpose(weights, (3, 2, 0, 1))

            # Save reshaped kernel weights and bias
            np.save(os.path.join(weights_dir, f"{layer.name}_weights.npy"), reshaped_weights)
            np.save(os.path.join(weights_dir, f"{layer.name}_bias.npy"), bias)

        # If batch normalization, save running mean and variance separately
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, moving_mean, moving_variance = layer.get_weights()

        # Save gamma, beta, moving mean, and moving variance
            np.save(os.path.join(weights_dir, f"{layer.name}_gamma.npy"), gamma)
            np.save(os.path.join(weights_dir, f"{layer.name}_beta.npy"), beta)
            np.save(os.path.join(weights_dir, f"{layer.name}_rmean.npy"), moving_mean)
            np.save(os.path.join(weights_dir, f"{layer.name}_rvar.npy"), moving_variance)

print("Weights saved in 'weights' folder.")
