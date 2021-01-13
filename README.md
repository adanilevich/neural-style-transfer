# neural-style-transfer

Transfer image style to your images with Neural Style Transfer using Tensorflow2.

## 1. Installation and start

Local or Google Colab execution is possible. Google colab execution is recommended due
to Colab GPU support (reduce calculation time).

### Google Colab

- copy ``nst_gui_colab.ipynb`` to your
- execute first cell: this git repository will be cloned to your Colab Linux instance
- Colab should provide all required dependencies. In case dependencies are missing
  execute in a Colab notebook cell: 
  > ! pip install -r requirements.txt
- upon execution of second cell, a jupyter widget GUI will appear in your notebook

### Local Execution

- requires python and Jupyter Lab or Notebook installed
- clone git repository
- intall dependencies:
  > ! pip install -r requirements.txt
- start ``nst_gui_local.ipynb`` and execute first cell

## 2. Usage

The widget allows you to choose a style image select from ``images/`` and a content image
and choose all neural net paramters of the underlying VGG19 network to tune for best
results. Following parameters can be controlled via GUI:
- number of epochs - should be somewhere between 100 and 1000
- learning rate - between 0.01 and 1
- content weight in the loss function: between 0.01 and 100
- style weight in the loss function: between 0.01 and 100
- VGG19 layer (and their respective weights) to use for content and style losses

Clicking ``generate`` will start image generation. Try experimenting with layer selection
and layer weights for optimal results.

See original paper in ``docs/`` for additional information

## 3. Implementation details

- VGG19 MaxPooling is replaced with AvgPooling for smoother results
- content/style layer weights are normalized to 1 (content/style separately)
- loss function calculates L1 loss (i.e. mean absolute error, not mean squared error),
both for content and style sub-losses
- after each iteration image values are clipped to [0, 1]