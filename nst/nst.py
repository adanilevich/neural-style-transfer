import tensorflow as tf
from tensorflow.keras.applications import VGG19
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
#from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np


class NSTModel:

    def __init__(self, content_layers: list = None, style_layers: list = None,
                 base_model: tf.keras.models.Model = None):

        if base_model is None:
            base_model = VGG19(include_top=False)

        if content_layers is None:
            content_layers = ['block4_conv2']

        if style_layers is None:
            style_layers = [
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


def init_result(result_shape):
    result_image = tf.Variable(
        initial_value=tf.random.uniform(shape=result_shape, minval=0, maxval=255),
        trainable=True
    )

    return result_image


def calc_content_loss(content_contents: list, result_contents: list) -> tf.Tensor:
    """
    Calculates content loss between processed target image and processed content image
    as meas squared error.

    Args:
        content_contents: List of tf.Tensors. List of content layers outputs for content image
            processed through nst model.
        result_contents: List of tf.Tensors. List of content layers outputs for
            result target image processed through nst model.

    Returns:
        loss: content loss
    """

    mse = tf.keras.losses.MeanSquaredError()
    losses = [mse(cont, res) for cont, res in zip(content_contents, result_contents)]
    loss = tf.add_n(losses)/len(losses)

    return loss


def calc_style_loss(style_style_outputs: list,
                    result_style_outputs: list) -> tf.Tensor:
    """
    Calculates style loss between processed target image and processed style image
    as meas squared error.

    Args:
        style_style_outputs: List of tf.Tensors. List of style layer
            outputs of style image processed through nst model
        result_style_outputs: List of tf.Tensors. List of style layer outputs
            of result image processed through nst model

    Returns:
        loss: style loss
    """

    def calc_style(style_layer_output: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(style_layer_output)
        style = tf.einsum('ijkl,ijkm->ilm', style_layer_output, style_layer_output)
        style = style / tf.cast(shape[1] * shape[2], tf.float32)

        return style

    style_styles = [calc_style(slo) for slo in style_style_outputs]
    result_styles = [calc_style(slo) for slo in result_style_outputs]

    mse = tf.keras.losses.MeanSquaredError()
    losses = [mse(st, res) for st, res in zip(style_styles, result_styles)]
    loss = tf.add_n(losses)/len(losses)

    return loss


def calc_total_loss(content_content_outputs: list, style_style_outputs: list,
                    result_content_outputs: list, result_style_outputs: list,
                    weights: dict) -> tf.Tensor:
    """
    Caculates total loss as weighted sum of content loss and style loss

    Args:
        content_content_outputs: List of tf.Tensors: list of content layer ouptuts
            of content image processed through model
        style_style_outputs: List of tf.Tensors: list of style layer ouptuts
            of style image processed through model
        result_content_outputs: List of tf.Tensors: list of content layer ouptuts
            of result image processed through model
        result_style_outputs: List of tf.Tensors: list of style layer ouptuts
            of result image processed through model
        weights: dict with keys 'content_weight', 'style_weight'

    Returns:
        loss: style_loss
    """

    content_loss = calc_content_loss(content_content_outputs, result_content_outputs)
    style_loss = calc_style_loss(style_style_outputs, result_style_outputs)

    content_weight = weights['content_weight']
    style_weight = weights['style_weight']

    total_loss = style_loss * style_weight + content_loss * content_weight

    return total_loss


@tf.function()
def calculate_gradients(content: tf.Variable, style: tf.Variable, result: tf.Variable,
                        loss_function: tf.keras.losses.Loss, model: tf.keras.Model,
                        weights: dict) -> list:
    """
    Calculates loss and its gradients with respect to target image

    Args:
        content: content image before processing through model
        style: style image before processing through model
        result: result image before processing through model
        loss_function: loss function to appply to content, style, result
        model: keras model for NST
        weights: dict with keys 'content_weight', 'style_weight'

    Returns:
        loss_value: total loss value (content and style) between three images
        grads: gradients of loss with respect to input image
    """

    with tf.GradientTape() as tape:
        content_content_outputs = model.process(content)[0]
        style_style_outputs = model.process(style)[1]
        result_content_ouputs = model.process(result)[0]
        result_style_outputs = model.process(result)[1]

        loss_function_parameters = {
            'content_content_outputs': content_content_outputs,
            'style_style_outputs': style_style_outputs,
            'result_content_outputs': result_content_ouputs,
            'result_style_outputs': result_style_outputs,
            'weights': weights
        }
        loss_value = loss_function(**loss_function_parameters)

    grads = tape.gradient(loss_value, result)

    return loss_value, grads


def normalize_image(image):
    """
    Rescales np array to range 0..1
    """

    max_ = np.max(image)
    min_ = np.min(image)
    image = (image - min_) / (max_ - min_)

    return image


def generate_nst(content: tf.Variable, style: tf.Variable, model: NSTModel,
                 epochs: int, lr: float, weights: dict, start_from_content=True) -> list:
    """
    Performs optimization to return generated image

    Args:
        content: content image before processing through model
        style: style image before processing through model
        model: keras model for NST
        epochs: number of fit iterations
        lr: learning rate (approx. 1)
        weights: dict with keys 'content_weight', 'style_weight'
        start_from_content: if True, initial image will be set to content

    Returns:
        trained image: resulting nst image as np array of shape (x, y, 3), scaled to 0-1;
            here, shape is the same as input model shape
        losses: list of loss values for each training iteration
    """

    if start_from_content:
        result = content
    else:
        result_shape = tuple([1]) + model.input_shape
        result = tf.Variable(
            initial_value=tf.random.uniform(shape=result_shape, minval=0, maxval=255),
            trainable=True
        )

    optimizer = tf.keras.optimizers.Adam(lr=lr)
    losses = []

    for step in range(epochs):

        if step % 10 == 0:
            print('Step:', step)

        gradient_parameters = {
            'content': content,
            'style': style,
            'result': result,
            'model': model,
            'loss_function': calc_total_loss,
            'weights': weights
        }
        loss, grads = calculate_gradients(**gradient_parameters)

        losses.append(loss)
        optimizer.apply_gradients([(grads, result)])

    trained_image = result.numpy().reshape(model.input_shape)
    trained_image = normalize_image(trained_image)

    return trained_image, losses