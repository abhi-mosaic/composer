from composer import Trainer
from composer import ComposerModel
import torch.nn as nn

class BasicComposerModel(ComposerModel):
    def __init__(self, n_layers, d_model):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.net = nn.Sequential(*[nn.Linear(d_model, d_model) * n_layers])

    def forward(self, x):
        return self.net(x)

    def loss(self, outputs, batch):
        return 

