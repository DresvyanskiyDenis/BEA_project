from typing import List, Optional

import torch
from torch import nn
from torchinfo import summary

from pytorch_utils.layers.attention_layers import Transformer_layer

class Seq2one_model_unwindowed(nn.Module):

    def __init__(self, input_size:int,
                 num_classes:List[int], transformer_num_heads:int,
                 num_transformer_layers:Optional[int]=1):
        super(Seq2one_model_unwindowed, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.transformer_num_heads = transformer_num_heads
        self.num_transformer_layers = num_transformer_layers

        # create transformer layer for multimodal cross-fusion
        if num_transformer_layers == 1:
            self.transformer_layers = Transformer_layer(input_dim=input_size,
                                                        num_heads=transformer_num_heads,
                                                        dropout=0.2,
                                                        positional_encoding=True)
        else:
            self.transformer_layers = nn.ModuleList([Transformer_layer(input_dim=input_size,
                                                                     num_heads=transformer_num_heads,
                                                                     dropout=0.2,
                                                                     positional_encoding=True) for _ in range(num_transformer_layers)])



        self.start_dropout = nn.Dropout(0.2)
        # get rid of timesteps
        self.max_pool_after_transformer = nn.AdaptiveMaxPool1d(1)
        self.average_pool_after_transformer = nn.AdaptiveAvgPool1d(1)
        self.embeddings = nn.Linear(input_size*2, input_size//2)
        self.batch_norm = nn.BatchNorm1d(input_size//2)
        self.activation_embeddings = nn.Tanh()
        self.end_dropout = nn.Dropout(0.2)

        # create classifier
        self.classifiers = nn.ModuleList([nn.Linear(input_size//2, num_class) for num_class in num_classes])

    def forward(self, x):
        # input shape (batch_size, seq_len, num_features)
        # transformer layers
        if self.num_transformer_layers == 1:
            x = self.transformer_layers(key=x, value=x, query=x)
        else:
            for i in range(self.num_transformer_layers):
                x = self.transformer_layers[i](key=x, value=x, query=x)
        # dropout after transformer layers
        x = self.start_dropout(x)
        # permute x to have the shape (batch_size, num_features, seq_len)
        x = x.permute(0, 2, 1)
        # max pool and average pool
        x_max = self.max_pool_after_transformer(x)
        x_avg = self.average_pool_after_transformer(x)
        # concatenate max and average pooled features
        x = torch.cat([x_max.squeeze(), x_avg.squeeze()], dim=-1)
        # one more linear layer
        x = self.embeddings(x)
        x = self.batch_norm(x)
        x = self.activation_embeddings(x)
        x = self.end_dropout(x)
        # classifiers
        classifiers_outputs = [classifier(x) for classifier in self.classifiers]

        return classifiers_outputs




if __name__ == "__main__":
    model = Seq2one_model_unwindowed(input_size=256, num_classes=[3,3,3], transformer_num_heads=8, num_transformer_layers=1)
    summary(model, input_size=(2, 10, 256), device="cpu")