import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

from nst.nst import NSTModel, generate_nst


class NSTGui:
    """
    Jupyter GUI for Neural Style Transfer. Supports selection of content and style
    images, training and plotting.
    """

    def __init__(self):
        # DEFAULT VALUES
        parent_path = Path(__file__).parent.parent
        image_path = parent_path/'images'
        self._content_image_path = parent_path/'images/content_dog.jpg'
        self._style_image_path = parent_path/'images/style_kandinsky_7.jpg'
        self._content = mpimg.imread(self._content_image_path)
        self._style = mpimg.imread(self._style_image_path)
        self._result = mpimg.imread(self._content_image_path)
        self._nst_model = None

        # IMAGE SELECTION

        self._content_selection = widgets.Dropdown(
            options = [f.name for f in image_path.iterdir()
                       if f.is_file() and 'content' in f.name],
            value = self._content_image_path.name,
            description = 'Content'
        )

        self._style_selection = widgets.Dropdown(
            options = [f.name for f in image_path.iterdir()
                       if f.is_file() and 'style' in f.name],
            value = self._style_image_path.name,
            description = 'Style'
        )

        self._display_selection_button = widgets.Button(description='Display')
        self._display_selection_button.on_click(self._click_display_selection)

        # TRAINING PARAMETERS
        self._epoch_selection = widgets.BoundedIntText(
            value=1000,
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
            value=10000,
            min=0.0,
            max=10000,
            description='Content Weight:',
            style={'description_width': '40%'}
        )

        self._style_weight_selection = widgets.BoundedFloatText(
            value=0.01,
            min=0.0,
            max=10000,
            description='Style Weight:',
            style={'description_width': '40%'}
        )

        # GENERATE IMAGE AND SAVE RESULTS BUTTONS
        self._generate_button = widgets.Button(description='Generate!')
        self._generate_button.on_click(self._click_generate)
        self._progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=self._epoch_selection.value,
            bar_style='',
            orientation='horizontal',
            layout=widgets.Layout(width='97%')
        )

        self._save_results_button = widgets.Button(description='Save Parameters')
        self._save_results_button.on_click(self._click_save_parameters)

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

        self._content_layer_selection  = widgets.VBox(
            [widgets.Checkbox(value=False, description=val) for val in content_options]
        )
        self._content_layer_selection.children[-3].value = True

        self._style_layer_selection  = widgets.VBox(
            [widgets.Checkbox(value=True, description=val) for val in style_options]
        )

        self._pooling_selection = widgets.RadioButtons(
            value='MaxPooling',
            options=['MaxPooling', 'AvgPooling'],
            description='Poolig Layers',
            layout={'width': 'max-content'}
        )

        # CREATE OUTPUTS
        self._image_output = widgets.Output()
        self._text_output = widgets.Output()

        # COMPOSE GUI
        self._gui = self._compose_gui()

        # PLOT DEFAULTS
        self._plot_images()

    def _plot_images(self):
        """
        Plots provided images to image ouput. Using pyplot

        Args:
            content, style, result: np.array of shape (width, len, channels)
        """
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
        self._content_image_path = parent_path/f'images/{self._content_selection.value}'
        self._content = mpimg.imread(self._content_image_path)
        self._style_image_path = parent_path / f'images/{self._style_selection.value}'
        self._style = mpimg.imread(self._style_image_path)
        self._result = mpimg.imread(self._content_image_path)
        self._plot_images()

    def _click_generate(self, b: widgets.Button):

        self._progress_bar.max = self._epoch_selection.value

        content_layers = [l.description for l in self._content_layer_selection.children
                          if l.value]

        style_layers = [l.description for l in self._style_layer_selection.children
                          if l.value]

        pooling = 'AvgPooling'

        self._nst_model = NSTModel(content_layers, style_layers, pooling)

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
            print(content_layers)
            print(style_layers)

            print_parameters = {
                k: v for k, v in generator_parameters.items()
                if k not in ['content', 'style', 'model', 'callback']
            }
            print('Starting generation with following parameters:\n')
            [print(f'{k}:', v) for k, v in print_parameters.items()]
            print('content layers:', self._nst_model.content_layers)
            print('style layers:', self._nst_model.style_layers)
            print('\n')

            self._result, losses_ = generate_nst(**generator_parameters)

            plt.plot(losses_)
            plt.show()

        self._plot_images()

    def _click_save_parameters(self, b: widgets.Button):
        """
        Saves selected parameters to 'saved_parameters.csv'
        """
        content_layers = [l.description for l in self._content_layer_selection.children
                          if l.value]

        style_layers = [l.description for l in self._style_layer_selection.children
                          if l.value]

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
            widgets.VBox([
                self._display_selection_button,
            ], layout=layout_padding),
        ], layout=layout_boxes)
        image_selection.layout.width = '80%'

        training_selection = widgets.VBox([
            self._epoch_selection,
            self._lr_selection,
            self._content_weight_selection,
            self._style_weight_selection
        ], layout=layout_boxes)


        layer_selection = widgets.HBox([
            widgets.Label('Content Layers:'),
            self._content_layer_selection,
            widgets.Label('Style Layers:'),
            self._style_layer_selection,
            #self._pooling_selection,
            widgets.VBox([
                self._generate_button,
                self._progress_bar,
                self._save_results_button
            ])
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
