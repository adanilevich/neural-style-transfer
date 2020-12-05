import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np

from nst.nst import normalize_image, NSTModel, generate_nst


class NSTGui():
    """
    Jupyter GUI for Neural Style Transfer. Supports selection of content and style
    images, training and plotting.
    """

    def __init__(self):

        # DEFAULT VALUES
        self._content_image_path = None
        self._style_image_path = None
        self._nst_model = None

        # IMAGE SELECTION
        self._selected_content_image = widgets.Label(value=str(self._content_image_path))
        self._selected_style_image = widgets.Label(value=str(self._style_image_path))
        self._select_content_image_button = \
            widgets.Button(description='Select content image')
        self._select_content_image_button.on_click(self._click_select_images_button)
        self._select_style_image_button = \
            widgets.Button(description='Select style image')
        self._select_style_image_button.on_click(self._click_select_images_button)

        # TRAINING PARAMETERS
        self._epoch_selection = widgets.BoundedIntText(
            value=100,
            min=10,
            max=10000,
            step=10,
            description='Epochs:',
            #style = {'description_width': 'initial'}
            style={'description_width': '40%'}
        )

        self._lr_selection = widgets.BoundedFloatText(
            value=0.1,
            min=0.0001,
            max=10,
            description='LR:',
            style={'description_width': '40%'}
        )

        self._content_weight_selection = widgets.BoundedFloatText(
            value=1,
            min=0.0001,
            max=10000,
            description='Content Weight:',
            style={'description_width': '40%'}
        )

        self._style_weight_selection = widgets.BoundedFloatText(
            value=1,
            min=0.0001,
            max=10000,
            description='Style Weight:',
            style={'description_width': '40%'}
        )

        # GENERATE IMAGE BUTTON
        self._generate_button = widgets.Button(description='Generate New Image!')
        self._generate_button.on_click(self._click_generate)

        # CREATE OUTPUTS
        self._image_selection_output = widgets.Output()
        self._nst_result_output = widgets.Output()
        self._text_output = widgets.Output()

        # COMPOSE GUI
        self._gui = self._compose_gui()

    def _click_select_images_button(self, b: widgets.Button):

        self._content_image_path = 'images/content_carina_2.jpg'
        self._style_image_path = 'images/style_kandinsky_7.jpg'

        self._selected_content_image.value = self._content_image_path
        self._selected_style_image.value = self._style_image_path

        content_raw = mpimg.imread(self._content_image_path)
        style_raw = mpimg.imread(self._style_image_path)

        self._image_selection_output.clear_output()
        with self._image_selection_output:
            fig, axes = plt.subplots(1, 2, figsize=(15, 10))
            axes[0].imshow(content_raw)
            axes[0].set_title('Content Image\n', fontsize=15)
            axes[0].axis('off') # disable axis lines, ticks, labels
            axes[1].imshow(style_raw)
            axes[1].set_title('Style Image \n', fontsize=15)
            axes[1].axis('off')  # disable axis lines, ticks, labels
            plt.show()


    def _click_generate(self, b: widgets.Button):

        input_shape = (224, 224, 3)
        content_pil = load_img(self._content_image_path, target_size=input_shape[:-1])
        content_np = img_to_array(content_pil)[np.newaxis, ...]
        style_pil = load_img(self._style_image_path, target_size=input_shape[:-1])
        style_np = img_to_array(style_pil)[np.newaxis, ...]

        content = tf.Variable(normalize_image(content_np), dtype=tf.float32)
        style = tf.Variable(normalize_image(style_np), dtype=tf.float32)

        if self._nst_model is None:
            self._nst_model = NSTModel()

        generator_parameters = {
            'epochs': self._epoch_selection.value,
            'model': self._nst_model,
            'content': content,
            'style': style,
            'lr': self._lr_selection.value,
            'weights': {
                'content_weight': self._content_weight_selection.value,
                'style_weight': self._style_weight_selection.value
            },
            'start_from_content': True
        }

        with self._text_output:
            self._text_output.clear_output()
            print_parameters = {
                k:v for k,v in generator_parameters.items()
                if k not in ['content', 'style', 'model', 'start_from_content']
            }
            print('Starting generation with following parameters:')
            [print(f'{k}:', v) for k, v in print_parameters.items()]

            self._trained_image, self._losses = generate_nst(**generator_parameters)

        with self._nst_result_output:
            self._nst_result_output.clear_output()
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.imshow(self._trained_image)
            ax.set_title('Result Image\n', fontsize=15)
            ax.axis('off') # disable axis lines, ticks, labels
            plt.show()



    def _compose_gui(self) -> widgets.Box:
        """
        Composes all widgets and returns final GUI.

        Returns:
            gui: final gui as widget
        """

        # DEFINE LAYOUTS
        layout_boxes = widgets.Layout(
            border='solid 1px black',
            margin='10px 10px 10px 0px', # spacing to other widgets; trbl
            padding='10px 10px 10px 10px' # spacing between border and widg inside; trbl
        )

        layout_padding = widgets.Layout(
            margin='0px 5px 0px 5px',  # spacing to other widgets
            padding='0px 0px 0px 0px'  # spacing between border and widgets inside
        )

        # DEFINE INPUT BOXES
        image_selection = widgets.HBox([
            widgets.VBox([
                self._select_content_image_button,
                self._select_style_image_button,
            ], layout=layout_padding),
            widgets.VBox([
                self._selected_content_image,
                self._selected_style_image,
            ], layout=layout_padding),
        ], layout=layout_boxes)
        image_selection.layout.width = '50%'

        training_selection = widgets.VBox([
                self._epoch_selection,
                self._lr_selection,
                self._content_weight_selection,
                self._style_weight_selection
        ], layout=layout_boxes)

        inputs = widgets.HBox([
            image_selection,
            training_selection,
        ])

        # GENERATE IMAGE BUTTON!
        generate_button_box = widgets.VBox([self._generate_button], layout=layout_boxes)
        generate_button_box.layout.width = '100%'

        # OUTPUT BOXES
        self._image_selection_output.layout = layout_boxes
        self._image_selection_output.layout.width = '98.9%'
        self._text_output.layout = layout_boxes
        self._text_output.width = '100%'
        self._nst_result_output.layout = layout_boxes
        self._nst_result_output.width = '100%'


        gui = widgets.VBox([
            inputs,
            generate_button_box,
            self._image_selection_output,
            self._nst_result_output,
            self._text_output
        ])

        return gui

    def draw(self):
        display(self._gui)