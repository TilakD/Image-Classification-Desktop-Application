# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, 
students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Training Examples
The following will train a densenet161 model executing on the GPU (Refer results_screenshot to see the obtained results. densenet161 and with 10 epochs, I am getting 95.8% accuracy.):

```python train.py flower_data --arch densenet161 --learning_rate 0.001 --gpu --epochs 10```

The following will train a vgg13 model executing on the GPU:

```python train.py flower_data --arch vgg13 --gpu --epochs 8```

## Prediction Examples.
The following will return the top 5 most likely classes using a pre-trained densenet161 model executing on the GPU and map categories to real names using a mapping file:

```python predict.py flower_data/test/66/image_05582.jpg checkpoint_CMD_APP.pth --gpu --top_k 5 --category_names cat_to_name.json```
