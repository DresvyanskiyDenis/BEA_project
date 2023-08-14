from typing import List, Optional

import torch
from torch import nn
from torchinfo import summary

from pytorch_utils.layers.attention_layers import Transformer_layer


class Seq2one_model(nn.Module):

    def __init__(self, input_size:int,
                 num_classes:List[int], transformer_num_heads:int, num_timesteps:int,
                 num_transformer_layers:Optional[int]=1):
        super(Seq2one_model, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.transformer_num_heads = transformer_num_heads
        self.num_timesteps = num_timesteps
        self.num_transformer_layers = num_transformer_layers

        # create transformer layer for multimodal cross-fusion

        self.transformer_layers = nn.ModuleList([Transformer_layer(input_dim=input_size,
                                                                     num_heads=transformer_num_heads,
                                                                     dropout=0.2,
                                                                     positional_encoding=True) for _ in range(num_transformer_layers)])


        # get rid of timesteps
        self.start_dropout = nn.Dropout(0.2)
        self.squeeze_layer_1 = nn.Conv1d(num_timesteps, 1, 1)
        self.squeeze_layer_2 = nn.Linear(input_size, input_size//2)
        self.batch_norm = nn.BatchNorm1d(input_size//2)
        self.activation_squeeze_layer = nn.Tanh()
        self.end_dropout = nn.Dropout(0.2)

        # create classifier
        self.classifiers = nn.ModuleList([nn.Linear(input_size//2, num_class) for num_class in num_classes])

    def forward(self, x):
        # transformer layers
        for i in range(self.num_transformer_layers):
            x = self.transformer_layers[i](key=x, value=x, query=x)
        # dropout after transformer layers
        x = self.start_dropout(x)
        # squeeze timesteps so that we have [batch_size, num_features]
        x = self.squeeze_layer_1(x)
        x = x.squeeze()
        # one more linear layer
        x = self.squeeze_layer_2(x)
        x = self.batch_norm(x)
        x = self.activation_squeeze_layer(x)
        x = self.end_dropout(x)
        # classifiers
        classifiers_outputs = [classifier(x) for classifier in self.classifiers]

        return classifiers_outputs



if __name__ == "__main__":
    x = torch.rand(10, 40, 256)
    y = torch.rand(10, 40, 1)
    model = Seq2one_model(input_size=256, num_classes=[3, 3, 3], transformer_num_heads=16, num_timesteps=40,
                          num_transformer_layers=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (10, 40, 256))
    output = model(x.to(device))

