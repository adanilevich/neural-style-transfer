from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import matplotlib.image as mpimg


# from tensorflow.keras.preprocessing.image import img_to_array

#TODO: use correct style layer weights -- see paper
#TODO: replace clipping by scaling for smooth gradients
class NSTModel():

    def __init__(self, content_layers: list = None, style_layers: list = None,
                 pooling: str = 'MaxPooling'):

        base_model = VGG19(include_top=False, weights='imagenet')
        base_model.trainable = False

        if pooling == 'AvgPooling':
            #print('Replacing MaxPooling by AvgPooling')
            base_model = replace_max_pooling(base_model)

        if content_layers is None:
            content_layers = ['block4_conv2']

        if style_layers is None:
            style_layers = [
                'block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
            ]

        self.content_layers = content_layers
        self.style_layers = style_layers

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
        Processes input image through neural network and returns style and content outputs

        Args:
            image: input image of shape (1, x, y, 3) and values in range [0, 1]

        Returns:
            result: dict with keys 'content', 'result', containing contents and styles
                of the image for each chosen layer - as tf.Tensor
        """


        image = image * 255.0
        image = preprocess_input(image)

        outputs = self.nst_model(image)
        contents = outputs['content']
        style_outputs = outputs['style']
        styles = [gram_matrix(style_output) for style_output in style_outputs]

        return {'content': contents, 'style': styles}


def replace_max_pooling(model):

    layers = [l for l in model.layers]

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
    content_contents = content_outputs['content']
    style_styles = style_outputs['style']
    result_contents = result_outputs['content']
    result_styles = result_outputs['style']
    content_weight = weights['content_weight']
    style_weight = weights['style_weight']

    # CONTENT LOSS
    content_losses = [tf.reduce_mean((cont - res)** 2)
                      for cont, res in zip(content_contents, result_contents)]
    content_loss = tf.add_n(content_losses) * content_weight / len(content_losses)

    # STYLE LOSS
    style_losses = [tf.reduce_mean((st - res) ** 2)
                    for st, res in zip(style_styles, result_styles)]
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

    original_shape = mpimg.imread(content_path).shape

    content = preprocess_image(content_path)
    style = preprocess_image(style_path)
    result = preprocess_image(content_path)
    result = tf.Variable(result)

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
        #result.assign(tf.clip_by_value(result, clip_value_min=0, clip_value_max=1))
        result = tf.Variable(normalize_image(result.numpy()), trainable=True, dtype=tf.float32)

    trained_image = postprocess_image(result, original_shape)

    return trained_image, losses


def preprocess_image(image_path) -> tf.Tensor:


    image = mpimg.imread(image_path).astype(np.float32)  # read to numpy array

    max_ = np.max(image.shape)
    scaling_factor = max_/512.0
    target_shape = [int(i/scaling_factor) for i in image.shape[:-1]] + [3]

    print(f'Preprocessing image {image_path.name} from {image.shape} to {target_shape}')
    image  = tf.image.resize(image, target_shape[:-1])
    image = image[np.newaxis, ...]  # add batch dimension
    image = image/255.0 # scale to [0, 1]

    return image


def normalize_image(image: np.array) -> np.array:
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
        image = normalize_image(image) * 255.0
        image = image - 128.0
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

    # # INVERTING VGG19 PREPROCESSING
    # image[:, :, 0] += 103.939
    # image[:, :, 1] += 116.779
    # image[:, :, 2] += 123.68
    # image = image[:, :, ::-1]

    image = tf.image.resize(image, original_shape[0:-1]).numpy()
    image = normalize_image(image) # normalize values to [0, 1]

    return image