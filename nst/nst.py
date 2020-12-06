from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import matplotlib.image as mpimg


# from tensorflow.keras.preprocessing.image import img_to_array


class NSTModel:

    def __init__(self, content_layers: list = None, style_layers: list = None,
                 base_model: tf.keras.models.Model = None):

        if base_model is None:
            base_model = VGG19(include_top=False)

        if content_layers is None:
            content_layers = ['block5_conv2']

        if style_layers is None:
            style_layers = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
            ]

        base_model.trainable = False
        self.input_shape = (224, 224, 3)
        self.content_layers = content_layers
        self.style_layers = style_layers

        content_outputs = []
        style_outputs = []

        for layer in base_model.layers:
            if layer.name in self.content_layers:
                content_outputs.append(layer.output)
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)

        outputs = content_outputs + style_outputs
        model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs)

        self.nst_model = model

    def process(self, image: tf.Variable) -> tf.Tensor:
        """
        Processes the input image through nst model and returns contents
        and styles

        Args:
            image: image to be processed as tf.Variable

        Returns:
            contents: list of resulting content outputs according to init paramters
            styles: list of resulting style outputs according to init parameters
        """

        contents = self.nst_model(image)[0:len(self.content_layers)]
        styles = self.nst_model(image)[len(self.content_layers):]

        return contents, styles


def calc_loss(content_outputs: list, style_outputs: list, result_outputs: list,
              weights: dict) -> tf.Tensor:
    """
    Caculates total loss as weighted sum of content loss and style loss

    Args:
        content_outputs: result of model.process(content_image)
        style_outputs: result of model.process(stlyle_image)
        result_outputs: result of model.process(result_image)
        weights: dict with keys 'content_weight', 'style_weight'

    Returns:
        loss: style_loss
    """

    # UNPACKING
    content_contents = content_outputs[0]
    style_styles = style_outputs[1]
    result_contents = result_outputs[0]
    result_styles = result_outputs[1]
    content_weight = weights['content_weight']
    style_weight = weights['style_weight']

    # CONTENT LOSS
    content_losses = [tf.reduce_mean((cont - res) ** 2)
                      for cont, res in zip(content_contents, result_contents)]
    content_loss = tf.add_n(content_losses)
    content_loss *= content_weight / len(content_losses)

    # STYLE LOSS
    def calc_style(style_layer_output: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(style_layer_output)
        style = tf.linalg.einsum('ijkl,ijkm->ilm', style_layer_output, style_layer_output)
        num_locations = tf.cast(shape[1] * shape[2] * shape[3], tf.float32)
        return style / (2 * num_locations)

    style_style_values = [calc_style(slo) for slo in style_styles]
    result_style_values = [calc_style(slo) for slo in result_styles]

    style_losses = [tf.reduce_mean((st - res) ** 2)
                    for st, res in zip(style_style_values, result_style_values)]
    style_loss = tf.add_n(style_losses)
    style_loss *= style_weight / len(style_losses)

    total_loss = style_loss + content_loss

    return total_loss


def generate_nst(content_path: Path, style_path: Path, model: NSTModel,
                 epochs: int, lr: float, weights: dict, callback=None) -> list:
    """
    Performs optimization to return generated image

    Args:
        content_path: path to content image
        style_path: path to style image
        model: keras model for NST
        epochs: number of fit iterations
        lr: learning rate (approx. 1)
        weights: dict with keys 'content_weight', 'style_weight'
        callback: progress bar callback

    Returns:
        trained image: resulting nst image as np array of shape (x, y, 3), scaled to 0-1;
            here, shape is the same as input model shape
        losses: list of loss values for each training iteration
    """

    original_shape = mpimg.imread(content_path).shape
    model_input_size = model.input_shape[0:-1]

    result = preprocess_image(content_path, model_input_size)
    content = preprocess_image(content_path, model_input_size)
    style = preprocess_image(style_path, model_input_size)

    optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.99, epsilon=1e-1)
    losses = []

    content_outputs = model.process(content)
    style_outputs = model.process(style)

    for step in range(epochs):

        callback.value = step + 1

        with tf.GradientTape() as tape:

            result_outputs = model.process(result)
            loss = calc_loss(content_outputs, style_outputs, result_outputs, weights)

        grads = tape.gradient(loss, result)

        losses.append(loss)
        optimizer.apply_gradients([(grads, result)])
        result = mid_process_image(result, clip_only=False)

    trained_image = postprocess_image(result, original_shape)

    return trained_image, losses


def normalize_image(image):
    """
    Rescales np array to range 0..1
    """

    max_ = np.max(image)
    min_ = np.min(image)
    image = (image - min_) / (max_ - min_)

    return image


def mid_process_image(image: tf.Variable, clip_only=False):
    """
    Used in each training step to adapt image to VGG specification before putting it
    through model processing

    Args:
        image: tf.Variable
        clip_only: if True, only processing is clipping of image values to [-115, 140]

    Returns:
        image: tf. Variable
    """

    if clip_only:
        image.assign(tf.clip_by_value(image, clip_value_min=-115.0, clip_value_max=140.0))
    else:
        image = image.numpy()
        print_channels('BEFORE PROCESSING', image)
        image = normalize_image(image) * 255.0
        print_channels('AFTER NORMALIZATION', image)

        image[:, :, :, 0] -= 103.939
        image[:, :, :, 1] -= 116.779
        image[:, :, :, 2] -= 123.68

        print_channels('AFTER VGG BALANCING', image)

        image = tf.Variable(image, dtype=tf.float32)

    return image

def print_channels(step, image):

    channel_map = {0: 'BLUE', 1: 'GREEN', 2: 'RED'}
    print(step)

    for channel in [0, 1, 2]:
        max = np.max(image[0, :, :, channel])
        min = np.min(image[0, :, :, channel])
        avg = np.mean(image[0, :, :, channel])
        print(f'CHANNEL: {channel_map[channel]}; MAX: {max}; MIN: {min}; AVG: {avg}')


def preprocess_image(image_path: Path, target_size: tuple) -> tf.Variable:
    """
    Transformns the image to VGG19 compatible format. In particular, image is resized,
    a batch dimension is added, channel values are normalized and RGB is converted to BGR
    Args:
        image_path: path to image
        target_size: input shape of VGG19, (224, 224, 3)

    Returns:
        image: image as tf.Variable for VGG19
    """

    image = mpimg.imread(image_path)  # read to numpy array
    print(f'Preprocessing image {image_path.name} from {image.shape} to {target_size}')
    image  = tf.image.resize(image, target_size)
    image = image[np.newaxis, ...]  # add batch dimension
    image = preprocess_input(image)
    image = tf.Variable(image, dtype=tf.float32)

    return image


def postprocess_image(image: tf.Tensor, original_shape: tuple) -> np.array:
    """
    Transforms created image from tf.Tensor to np. array. Image is resized to original
    size, batch dimension is removed, also VGG preprocessing is reverted. Output image
    is normalized to values [0, 1].

    Args:
        image: created image as tf.Tensor
        original_shape: original shape of the image to resize to

    Returns:
        image: image as np.array with values in [0, 1] and size of original image
    """

    image = image.numpy()
    print(f'\n Postprocessing result image from {image.shape} to {original_shape}')
    image = image.reshape(image.shape[1:])  # drop batch dimension

    # INVERTING VGG19 PREPROCESSING
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    image = image[:, :, ::-1]

    image = tf.image.resize(image, original_shape[0:-1]).numpy()
    image = normalize_image(image) # normalize values to [0, 1]

    return image