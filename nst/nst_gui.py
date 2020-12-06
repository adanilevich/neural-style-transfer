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
        #self._selected_content_image = widgets.Label(value=self._content_image_path.name)
        #self._selected_style_image = widgets.Label(value=self._style_image_path.name)

        #self._select_content_image_button = widgets.Button(description='Content Image')
        #self._select_content_image_button.on_click(self._click_select_images_button)

        #self._select_style_image_button = widgets.Button(description='Style Image')
        #self._select_style_image_button.on_click(self._click_select_images_button)

        self._content_selection = widgets.Dropdown(
            options = [f.name for f in image_path.iterdir() if f.is_file()],
            value = self._content_image_path.name,
            description = 'Content'
        )

        self._style_selection = widgets.Dropdown(
            options = [f.name for f in image_path.iterdir() if f.is_file()],
            value = self._style_image_path.name,
            description = 'Style'
        )

        self._display_selection_button = widgets.Button(description='Display')
        self._display_selection_button.on_click(self._click_display_selection)

        # TRAINING PARAMETERS
        self._epoch_selection = widgets.BoundedIntText(
            value=100,
            min=10,
            max=10000,
            step=10,
            description='Epochs:',
            # style = {'description_width': 'initial'}
            style={'description_width': '40%'}
        )

        self._lr_selection = widgets.BoundedFloatText(
            value=0.5,
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
            value=100,
            min=0.0001,
            max=10000,
            description='Style Weight:',
            style={'description_width': '40%'}
        )

        # GENERATE IMAGE BUTTON
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

    def _click_select_images_button(self, b: widgets.Button):

        display(widgets.FileUpload())

        self._text_output.clear_output()

        with self._text_output:
            print('Image selection currently disabled')

    def _click_generate(self, b: widgets.Button):

        self._progress_bar.max = self._epoch_selection.value

        if self._nst_model is None:
            self._nst_model = NSTModel()

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
                #self._select_content_image_button,
                #self._select_style_image_button,
                self._content_selection,
                self._style_selection,
                self._display_selection_button,
                self._generate_button
            ], layout=layout_padding),
            widgets.VBox([
                #self._selected_content_image,
                #self._selected_style_image,
                self._progress_bar
            ], layout=layout_padding),
        ], layout=layout_boxes)
        image_selection.layout.width = '80%'

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
