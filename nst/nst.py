from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import matplotlib.image as mpimg


# TODO: include style layer weights
# TODO: start from random
class NSTModel:

    def __init__(self, content_layers: dict = None, style_layers: dict = None,
                 pooling: str = 'AvgPooling'):
        """
        Initializes a neural network which will output content and style of a given image.
        Weights provided via 'content_layers' and 'style_layers' will be normalized to 1.

        Args:
            content_layers: dict with {vgg_layer_name: layer weight} for content layers
            style_layers: dict with {vgg_layer_name: layer weight} for style layers
            pooling: 'AvgPooling' or 'MaxPooling'. If 'AvgPooling', VGG19 Pooling layers
                will be replaced with average pooling.
        """

        # SET DEFAULTS AND NORMALIZE WEIGHT TO 1
        if content_layers is None:
            content_layers = {'block4_conv2': 1.0}

        if style_layers is None:
            style_layers = {
                'block1_conv1': 1.0,
                'block2_conv1': 1.0,
                'block3_conv1': 1.0,
                'block4_conv1': 1.0,
                'block5_conv1': 1.0,
            }

        sum_content_weights = sum([v for k, v in content_layers.items()])
        sum_style_weights = sum([v for k, v in style_layers.items()])
        content_layers = {k: v/sum_content_weights for k, v in content_layers.items()
                          if v != 0.}
        style_layers = {k: v/sum_style_weights for k, v in style_layers.items()
                        if v != 0.}

        self.content_layers = content_layers
        self.style_layers = style_layers

        print(self.content_layers)
        print(self.style_layers)

        # DOWNLOAD VGG19 AND CREATE NEW MODEL
        base_model = VGG19(include_top=False, weights='imagenet')
        base_model.trainable = False

        if pooling == 'AvgPooling':
            base_model = self._replace_max_pooling(base_model)

        content_outputs = []
        style_outputs = []

        self.content_layers = content_layers
        self.style_layers = style_layers

        for layer in base_model.layers:
            if layer.name in self.content_layers:
                content_outputs.append(layer.output)
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)

        outputs = {'content': content_outputs, 'style': style_outputs}
        model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs)

        self.nst_model = model

    def process(self, image: tf.Variable) -> dict:
        """
        Processes input image through neural network and returns style and
        content outputs, weights by layer weights defined in _init_.

        Args:
            image: input image of shape (1, x, y, 3) and values in range [0, 1]

        Returns:
            result: dict with keys 'content', 'result', containing contents and styles
                of the image for each chosen layer - as tf.Tensor
        """

        # PREPROCESS IMAGE WITH VGG PREPROCESSING
        image = image * 255.0
        image = preprocess_input(image)

        # ADD WEIGHTS TO OUTPUTS
        outputs = self.nst_model(image)
        content_weights = [v for k, v in self.content_layers.items()]
        contents = [w * o for w, o in zip(content_weights, outputs['content'])]

        style_weights = [v for k, v in self.style_layers.items()]
        styles_gram = [gram_matrix(style_output) for style_output in outputs['style']]
        styles = [w * s for w, s in zip(style_weights, styles_gram)]

        return {'content': contents, 'style': styles}


    def _replace_max_pooling(self, model):
        # replaces max pooling with average pooling in model and returns new model
        layers = [layer for layer in model.layers]

        x = layers[0].output
        for i in range(1, len(layers)):
            if 'MaxPooling' in str(layers[i]):
                x = tf.keras.layers.AveragePooling2D(2)(x)
            else:
                x = layers[i](x)

        new_model = tf.keras.Model(inputs=model.inputs, outputs=x)
        return new_model


def gram_matrix(style_layer_output: tf.Tensor) -> tf.Tensor:
    """
    Calculates Gram's matrix as inner product between different channels

    Args:
        style_layer_output: output of model layer of shape (i, j, k, l)

    Returns:
        gram: layer correlation matrix of shape (i, l, l)
    """

    shape = tf.shape(style_layer_output)
    style = tf.linalg.einsum('ijkl,ijkm->ilm', style_layer_output, style_layer_output)
    num_locations = tf.cast(shape[1] * shape[2], tf.float32)

    return style / num_locations


def calc_loss(content_outputs: dict, style_outputs: dict, result_outputs: dict,
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
    content_c = content_outputs['content']
    style_s = style_outputs['style']
    result_c = result_outputs['content']
    result_s = result_outputs['style']
    content_weight = weights['content_weight']
    style_weight = weights['style_weight']

    mean = tf.reduce_mean
    l1 = tf.abs  # mean absolute error
    # l2 = lambda x: x**2 # mean square error

    # CONTENT LOSS
    content_losses = [mean(l1(cont - res)) for cont, res in zip(content_c, result_c)]
    content_loss = tf.add_n(content_losses) * content_weight / len(content_losses)

    # STYLE LOSS
    style_losses = [mean(l1(st - res)) for st, res in zip(style_s, result_s)]
    style_loss = tf.add_n(style_losses) * style_weight / len(style_losses)

    return style_loss + content_loss


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

    print("\nStarting image processing...")
    original_shape = mpimg.imread(content_path).shape

    content = preprocess_image(content_path)
    style = preprocess_image(style_path)
    result = preprocess_image(content_path)
    result = tf.Variable(result)

    optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.99, epsilon=1e-1)
    losses = []

    content_outputs = model.process(content)
    style_outputs = model.process(style)

    print('\nStarting image generation...')
    for step in range(epochs):
        callback.value = step + 1

        with tf.GradientTape() as tape:
            result_outputs = model.process(result)
            loss = calc_loss(content_outputs, style_outputs, result_outputs, weights)

        grads = tape.gradient(loss, result)

        losses.append(loss)
        optimizer.apply_gradients([(grads, result)])
        result.assign(tf.clip_by_value(result, clip_value_min=0, clip_value_max=1))
        # result = tf.Variable(normalize_image(result.numpy()), trainable=True, dtype=tf.float32)
        # print('STEP:', np.max(result.numpy()), np.min(result.numpy()), np.mean(result.numpy()),
        #       np.mean(result.numpy()[0, :, :, 0]), np.mean(result.numpy()[0, :, :, 1]),
        #       np.mean(result.numpy()[0, :, :, 2]))

    trained_image = postprocess_image(result, original_shape)

    return trained_image, losses


def preprocess_image(image_path) -> tf.Tensor:
    image = mpimg.imread(image_path).astype(np.float32)  # read to numpy array

    max_ = np.max(image.shape)
    scaling_factor = max_ / 512.0
    target_shape = [int(i / scaling_factor) for i in image.shape[:-1]] + [3]

    print(f'Preprocessing {image_path.name} from {image.shape} to {target_shape}...')
    image = tf.image.resize(image, target_shape[:-1])
    image = image[np.newaxis, ...]  # add batch dimension
    image = image / 255.0  # scale to [0, 1]

    return image


def normalize_image(image: np.array) -> np.array:
    """
    Rescales np array to range 0..1
    """

    max_ = np.max(image)
    min_ = np.min(image)
    image = (image - min_) / (max_ - min_)

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
    print(f'\nPostprocessing result image from {image.shape} to {original_shape}')
    image = image.reshape(image.shape[1:])  # drop batch dimension

    # # INVERTING VGG19 PREPROCESSING
    # image[:, :, 0] += 103.939
    # image[:, :, 1] += 116.779
    # image[:, :, 2] += 123.68
    # image = image[:, :, ::-1]

    image = tf.image.resize(image, original_shape[0:-1]).numpy()
    image = normalize_image(image)  # normalize values to [0, 1]

    return image
