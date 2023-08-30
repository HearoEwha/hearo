from django.db import models
import torch
import torchvision
import torchvision.models.video as vmodels

class MyModel():
    def __init__(self):
        # Load the saved model
        model_path = "/Users/dohakim/HearoWeb/r2plus2d.pth"
        #self.model = torch.load(model_path, map_location=torch.device('cpu'))

        self.model = vmodels.r2plus1d_18(num_classes=15, pretrained=False)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)

        # Set device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()


# class MyModel():
#     # Load the saved model
#     model_path = "C:/Users/kdh30/Downloads/my_model.pth"
#     model = torch.load(model_path)
#
#     # Set device to use
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model.eval()



