import tensorflow as tf
from tensorflow.keras.applications import VGG19
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np


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


def init_result(result_shape):
    result_image = tf.Variable(
        initial_value=tf.random.uniform(shape=result_shape, minval=0, maxval=255),
        trainable=True
    )

    return result_image


def calc_total_loss(content_contents: list, style_styles: list,
                    result_contents: list, result_styles: list,
                    weights: dict) -> tf.Tensor:
    """
    Caculates total loss as weighted sum of content loss and style loss

    Args:
        content_contents: List of tf.Tensors: list of content layer ouptuts
            of content image processed through model
        style_styles: List of tf.Tensors: list of style layer ouptuts
            of style image processed through model
        result_contents: List of tf.Tensors: list of content layer ouptuts
            of result image processed through model
        result_styles: List of tf.Tensors: list of style layer ouptuts
            of result image processed through model
        weights: dict with keys 'content_weight', 'style_weight'

    Returns:
        loss: style_loss
    """

    content_losses = [tf.reduce_mean(cont - res)**2
                      for cont, res in zip(content_contents, result_contents)]
    content_loss = tf.add_n(content_losses) / len(content_losses)

    def calc_style(style_layer_output: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(style_layer_output)
        style = tf.linalg.einsum('ijkl,ijkm->ilm', style_layer_output, style_layer_output)
        num_locations = tf.cast(shape[1] * shape[2], tf.float32)

        return style/(num_locations)

    style_style_values = [calc_style(slo) for slo in style_styles]
    result_style_values = [calc_style(slo) for slo in result_styles]

    style_losses = [tf.reduce_mean(st - res)**2
                    for st, res in zip(style_style_values, result_style_values)]
    style_loss = tf.add_n(style_losses) / len(style_losses)

    content_weight = weights['content_weight']
    style_weight = weights['style_weight']

    total_loss = style_loss * style_weight + content_loss * content_weight

    return total_loss


@tf.function()
def calculate_gradients(content_outputs: list, style_outputs: list, result: tf.Variable,
                        loss_function: tf.keras.losses.Loss, model: tf.keras.Model,
                        weights: dict) -> list:
    """
    Calculates loss and its gradients with respect to target image

    Args:
        content_outpus: content image processes through model
        style_outputs: style image processed through model
        result: result image before processing through model
        loss_function: loss function to appply to content, style, result
        model: keras model for NST
        weights: dict with keys 'content_weight', 'style_weight'

    Returns:
        loss_value: total loss value (content and style) between three images
        grads: gradients of loss with respect to input image
    """

    with tf.GradientTape() as tape:

        result_outputs = model.process(result)

        loss_function_parameters = {
            'content_contents': content_outputs[0],
            'style_styles': style_outputs[1],
            'result_contents': result_outputs[0],
            'result_styles': result_outputs[1],
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


def generate_nst(content: np.array, style: np.array, model: NSTModel,
                 epochs: int, lr: float, weights: dict, callback=None) -> list:
    """
    Performs optimization to return generated image

    Args:
        content: content image before processing through model
        style: style image before processing through model
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

    original_shape = content.shape
    model_input_size = model.input_shape[0:-1]

    result = preprocess_image(content, model_input_size)
    result_keep = preprocess_image(content, model_input_size)
    content = preprocess_image(content, model_input_size)
    style = preprocess_image(style, model_input_size)

    print('\n')
    print(np.max(content.numpy()), np.min(content.numpy()))
    print(np.max(style.numpy()), np.min(style.numpy()))

    optimizer = tf.keras.optimizers.Adam(lr=lr, beta_1=0.99, epsilon=1e-1)
    losses = []

    content_outputs = model.process(content)
    style_outputs = model.process(style)

    for step in range(epochs):

        if step % (epochs / 100) == 0:
            callback.value = step
            callback.description = f'{int(100.0 * (step + 1) / epochs)}%'

        gradient_parameters = {
            'content_outputs': content_outputs,
            'style_outputs': style_outputs,
            'result': result,
            'model': model,
            'loss_function': calc_total_loss,
            'weights': weights
        }
        loss, grads = calculate_gradients(**gradient_parameters)

        losses.append(loss)
        optimizer.apply_gradients([(grads, result)])
        result.assign(tf.clip_by_value(result, clip_value_min=0.0, clip_value_max=1.0))

        print('MAX:', np.max(result.numpy()))
        print('MIN:', np.min(result.numpy()))
        print('GRAD:', tf.reduce_mean(grads**2).numpy())

    trained_image = result.numpy().reshape(model.input_shape)
    trained_image = tf.image.resize(trained_image, original_shape[0:-1]).numpy()
    trained_image = normalize_image(trained_image)

    return trained_image, losses


def preprocess_image(image: np.array, target_size: tuple) -> tf.Variable:
    #image = preprocess_input(image)
    image = image/255.0
    image = tf.image.resize(image, target_size)
    image = image.numpy()
    image = image[np.newaxis,...]
    image = tf.Variable(image, dtype=tf.float32)

    return image

