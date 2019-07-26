import torch

def init():
    global device
    global content_layers_default
    global style_layers_default
    global num_steps
    global style_weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    num_steps = 300
    style_weight = 1000000