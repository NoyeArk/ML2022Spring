import torch.nn as nn


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()

        # TODO: modify models's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(16, 8),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.05, inplace=False),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
