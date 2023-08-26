from django.db import models
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
import torchvision
import torchvision.models.video as vmodels

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class CombinedModel(nn.Module):
    def __init__(self, num_classes):
        super(CombinedModel, self).__init__()
        self.c3d = r3d_18(pretrained=True)  # 미리 학습된 C3D 모델 사용
        self.c3d.fc = nn.Identity()  # 마지막 FC 레이어 제거
        self.lstm = LSTMModel(input_size=512, hidden_size=128, num_layers=2, num_classes=num_classes)

    def forward(self, x):
        x = self.c3d(x)
        x = x.view(x.size(0), -1, 512)  # LSTM에 입력할 수 있도록 reshape
        out = self.lstm(x)
        return out

class MyModel():
    def __init__(self):
        # Load the saved model
        model_path = "/Users/song-yeojin/hearoWeb/r2plus2d.pth"
        #self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = vmodels.r2plus1d_18(num_classes=15, pretrained=False)
        #self.model = CombinedModel(num_classes = 15)
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



