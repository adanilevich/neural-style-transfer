import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

from nst.nst import NSTModel, generate_nst


class NSTGui:
    """
    Jupyter GUI for Neural Style Transfer. Supports selection of content and style
    images, training and plotting.

    Methods:
        draw: displays GUI as ipywidget
    """

    # IMAGE VARIABlES
    _content_image_path: Path
    _style_image_path: Path
    _content: np.array
    _style: np.array
    _result: np.array

    # MODEL VARIABLES
    _nst_model: NSTModel

    # GUI: IMAGE SELECTION
    _content_selection: widgets.Dropdown
    _style_selection: widgets.Dropdown
    _display_selection_button: widgets.Button

    # GUI: TRAINING PARAMETERS
    _epoch_selection: widgets.BoundedIntText
    _lr_selection: widgets.BoundedFloatText
    _content_weight_selection: widgets.BoundedFloatText
    _style_weight_selection: widgets.BoundedFloatText

    # GUI: MODEL ARCHITECTURE
    _content_layer_selection: widgets.VBox
    _style_layer_selection: widgets.VBox

    # GUI: OTHER
    _generate_button: widgets.Button
    _progress_bar: widgets.IntProgress
    _image_output: widgets.Output
    _text_output: widgets.Output
    _gui: widgets.VBox


    def __init__(self):
        # DEFAULT VALUES
        parent_path = Path(__file__).parent.parent
        image_path = parent_path / 'images'
        self._content_image_path = parent_path / 'images/content_dog.jpg'
        self._style_image_path = parent_path / 'images/style_kandinsky_7.jpg'
        self._content = mpimg.imread(self._content_image_path)
        self._style = mpimg.imread(self._style_image_path)
        self._result = mpimg.imread(self._content_image_path)
        self._nst_model = None

        # IMAGE SELECTION
        self._content_selection = widgets.Dropdown(
            options=[f.name for f in image_path.iterdir()
                     if f.is_file() and 'content' in f.name],
            value=self._content_image_path.name,
            description='Content',
            layout = widgets.Layout(width='80%')
        )

        self._style_selection = widgets.Dropdown(
            options=[f.name for f in image_path.iterdir()
                     if f.is_file() and 'style' in f.name],
            value=self._style_image_path.name,
            description='Style',
            layout=widgets.Layout(width='80%')
        )

        self._display_selection_button = widgets.Button(
            description='Display',
            layout=widgets.Layout(width='20%')
        )
        self._display_selection_button.on_click(self._click_display_selection)

        # TRAINING PARAMETERS
        self._epoch_selection = widgets.BoundedIntText(
            value=100,
            min=5,
            max=10000,
            step=10,
            description='Epochs:',
            # style = {'description_width': 'initial'}
            style={'description_width': '40%'}
        )

        self._lr_selection = widgets.BoundedFloatText(
            value=0.02,
            min=0.0001,
            max=10,
            description='LR:',
            style={'description_width': '40%'}
        )

        self._content_weight_selection = widgets.BoundedFloatText(
            value=1,
            min=0.0,
            max=10000,
            description='Content Weight:',
            style={'description_width': '40%'}
        )

        self._style_weight_selection = widgets.BoundedFloatText(
            value=1,
            min=0.0,
            max=10000,
            description='Style Weight:',
            style={'description_width': '40%'}
        )

        # LAYER SELECTION
        content_options = [
            'block1_conv2',
            'block2_conv2',
            'block3_conv2',
            'block4_conv2',
            'block5_conv2',
        ]

        style_options = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
        ]

        self._content_layer_selection = widgets.VBox([
            widgets.FloatText(
                layout=widgets.Layout(width='90%'),
                description=val,
                style={'description_width': '70%'}
            )
            for val in content_options
        ], layout=widgets.Layout(width='25%'))
        self._content_layer_selection.children[-3].value = 1

        self._style_layer_selection = widgets.VBox([
            widgets.FloatText(
                layout=widgets.Layout(width='90%'),
                style={'description_width': '70%'},
                description=v,
                value=1
            )
            for v in style_options
        ], layout=widgets.Layout(width='25%'))

        # self._style_layer_selection = widgets.VBox(
        #     [widgets.Checkbox(value=True, description=val) for val in style_options]
        # )

        # GENERATE IMAGE AND SAVE RESULTS BUTTONS
        self._generate_button = widgets.Button(
            description='Generate!',
            layout=widgets.Layout(width='90%')
        )
        self._generate_button.on_click(self._click_generate)
        self._progress_bar = widgets.IntProgress(
            max=self._epoch_selection.value,
            layout=widgets.Layout(width='90%')
        )
        self._save_results_button = widgets.Button(
            description='Save Parameters',
            layout=widgets.Layout(width='90%')
        )
        self._save_results_button.on_click(self._click_save_parameters)

        # CREATE OUTPUTS AND COMPOSE GUI
        self._image_output = widgets.Output()
        self._text_output = widgets.Output()
        self._gui = self._compose_gui()
        self._plot_images()

    def _plot_images(self):

        self._image_output.clear_output()

        with self._image_output:
            fig, axes = plt.subplots(1, 3, figsize=(30, 30))

            axes[0].imshow(self._content)
            axes[0].set_title('Content Image\n', fontsize=15)
            axes[0].axis('off')  # disable axis lines, ticks, labels

            axes[1].imshow(self._style)
            axes[1].set_title('Style Image \n', fontsize=15)
            axes[1].axis('off')  # disable axis lines, ticks, labels

            axes[2].imshow(self._result)
            axes[2].set_title('Result Image \n', fontsize=15)
            axes[2].axis('off')  # disable axis lines, ticks, labels

            plt.show()

    def _click_display_selection(self, b: widgets.Button):

        parent_path = Path(__file__).parent.parent
        self._content_image_path = parent_path / f'images/{self._content_selection.value}'
        self._content = mpimg.imread(self._content_image_path)
        self._style_image_path = parent_path / f'images/{self._style_selection.value}'
        self._style = mpimg.imread(self._style_image_path)
        self._result = mpimg.imread(self._content_image_path)
        self._plot_images()

    def _click_generate(self, b: widgets.Button):
        self._progress_bar.max = self._epoch_selection.value

        content_layers = {layer.description: layer.value for layer in
                          self._content_layer_selection.children if layer.value != 0.}
        style_layers = {layer.description: layer.value for layer in
                        self._style_layer_selection.children if layer.value != 0.}

        self._nst_model = NSTModel(content_layers, style_layers, pooling='AvgPooling')

        generator_parameters = {
            'epochs': self._epoch_selection.value,
            'model': self._nst_model,
            'content_path': self._content_image_path,
            'style_path': self._style_image_path,
            'lr': self._lr_selection.value,
            'weights': {
                'content_weight': self._content_weight_selection.value,
                'style_weight': self._style_weight_selection.value
            },
            'callback': self._progress_bar
        }

        with self._text_output:
            self._text_output.clear_output()
            print('Initialized NST model with following layers and weights:')
            print(f'\tcontent layers: {self._nst_model.content_layers}')
            print(f'\tstyle layers: {self._nst_model.style_layers}')

            print_parameters = {
                k: v for k, v in generator_parameters.items()
                if k not in ['content', 'style', 'model', 'callback']
            }
            print('\nStarting generation with following parameters:')
            [print(f'\t{k}:', v) for k, v in print_parameters.items()]

            self._result, losses_ = generate_nst(**generator_parameters)

            print('\nPlotting loss function...')
            plt.plot(losses_)
            plt.title('Loss function')
            plt.show()

        self._plot_images()

    def _click_save_parameters(self, b: widgets.Button):
        """
        Saves selected parameters to 'saved_parameters.csv'
        """
        content_layers = [layer.description for layer in
                          self._content_layer_selection.children if layer.value]

        style_layers = [layer.description for layer in
                        self._style_layer_selection.children if layer.value]

        parameters = [
            f'content: {self._content_image_path.name}',
            f'style: {self._style_image_path.name}',
            f'epoch: {self._epoch_selection.value}',
            f'lr: {self._lr_selection.value}',
            f'content_weight: {self._content_weight_selection.value}',
            f'style_weight: {self._style_weight_selection.value}',
            f'content_layers: {content_layers}',
            f'style_layers: {style_layers}',
        ]

        with open('saved_parameters.txt', 'a') as file:
            file.write(f"{'; '.join(parameters)}\n")

    def _compose_gui(self) -> widgets.Box:
        """
        Composes all widgets and returns final GUI.

        Returns:
            gui: final gui as widget
        """

        # DEFINE LAYOUTS
        layout_boxes = widgets.Layout(
            border='solid 1px black',
            margin='10px 10px 10px 0px',  # spacing to other widgets; trbl
            padding='10px 10px 10px 10px'  # spacing between border and widg inside; trbl
        )

        layout_padding = widgets.Layout(
            margin='0px 5px 0px 5px',  # spacing to other widgets
            padding='0px 0px 0px 0px'  # spacing between border and widgets inside
        )

        # DEFINE INPUT BOXES
        image_selection = widgets.HBox([
            widgets.VBox([
                self._content_selection,
                self._style_selection,
            ], layout=layout_padding),
            self._display_selection_button,
        ], layout=layout_boxes)
        image_selection.layout.width = '80%'

        training_selection = widgets.VBox([
            self._epoch_selection,
            self._lr_selection,
            self._content_weight_selection,
            self._style_weight_selection
        ], layout=layout_boxes)

        layer_selection = widgets.HBox([
            widgets.Label('Content Layers:', layout = widgets.Layout(width='15%')),
            self._content_layer_selection,
            widgets.Label('Style Layers:', layout = widgets.Layout(width='15%')),
            self._style_layer_selection,
            widgets.VBox([
                self._generate_button,
                self._progress_bar,
                self._save_results_button
            ], layout = widgets.Layout(width='20%'))
        ], layout=layout_boxes)

        inputs = widgets.VBox([
            widgets.HBox([
                image_selection,
                training_selection,
            ]),
            layer_selection
        ])

        # OUTPUT BOXES
        self._image_output.layout = layout_boxes
        self._image_output.layout.width = '98.9%'
        self._text_output.layout = layout_boxes
        self._text_output.width = '100%'

        gui = widgets.VBox([
            inputs,
            self._image_output,
            self._text_output
        ])

        return gui

    def draw(self):
        display(self._gui)
