import torch
import torch.nn as nn

class RectifiedFlow(nn.Module):

    def __init__(self, n) -> None:
        super().__init__()

        nodes = [64,64,64]

        self.first_layer = nn.Linear(n+1, nodes[0])
        self.first_hidden_layer = nn.Linear(nodes[0],nodes[1])
        self.second__hidden_layer = nn.Linear(nodes[1],nodes[2])
        self.final_layer = nn.Linear(nodes[2],n)
    

    def forward(self, x, t):
        x = torch.cat((x,t),dim=-1).to(torch.float)
        x = self.first_layer(x)
        x = torch.tanh(x)
        x = self.first_hidden_layer(x)
        x = torch.tanh(x)
        x = self.second__hidden_layer(x)
        x = torch.tanh(x)
        x = self.final_layer(x)

        return x

