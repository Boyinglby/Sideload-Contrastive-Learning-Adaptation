from ultralytics import YOLO
import torch
import random
import numpy as np

def load_layer_weights(from_layers, to_layers):
    for model_layer, yolo_layer in zip(from_layers, to_layers):
        model_layer_state_dict = model_layer.state_dict()
        yolo_layer_state_dict = yolo_layer.state_dict()

        # Update yolo_layer_state_dict with weights from model_layer_state_dict
        for key in yolo_layer_state_dict.keys():
            if key in model_layer_state_dict and model_layer_state_dict[key].shape == yolo_layer_state_dict[key].shape:
                yolo_layer_state_dict[key] = model_layer_state_dict[key]
            else:
                # print(f'layer: {model_layer} not matching')
                print(model_layer_state_dict[key].shape)
                print(yolo_layer_state_dict[key].shape)
 
        # Load the updated state dictionary back into the yolo layer
        yolo_layer.load_state_dict(yolo_layer_state_dict)

# --------------------Load pretrained CL as side CNN --------------------
# Load the state dictionary

model_CL = torch.load('sideCNN_CLpretrained.pt')

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

side_yolo = YOLO("./yolo_config/yolo11n-sideCNN_yolobackbone_SE.yaml")
yolo11n_model = YOLO('yolo11n.pt')

# Load weights for these layers
model_CL_layers = list(model_CL.children())[:9]

side_cnn_layers = list(side_yolo.model.model.children())[1:10]

load_layer_weights(model_CL_layers, side_cnn_layers)



# --------------Load coco pretrained yolo11n weights for backbone layers 1-11 and head layers 18-30-----------------
yolo_backbone_layers = list(yolo11n_model.model.model.children())[:9]
yolo_neck_layers = list(yolo11n_model.model.model.children())[9:11]
yolo_head_layers = list(yolo11n_model.model.model.children())[11:]

side_backbone_layers = list(side_yolo.model.model.children())[10:19] 
side_neck_layers = list(side_yolo.model.model.children())[21:23]
side_head_layers = list(side_yolo.model.model.children())[23:]

load_layer_weights(yolo_backbone_layers, side_backbone_layers)
load_layer_weights(yolo_head_layers, side_head_layers)
load_layer_weights(yolo_neck_layers, side_neck_layers)


side_yolo.save('sideload_SE_gating.pt')


