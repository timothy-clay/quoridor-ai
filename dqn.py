import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        
        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # forward pass steps
        self.fc = nn.Sequential(
            nn.Linear(128 + 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, state, prev_move_onehot):

        # perform convolutions on game state and flatten result
        conv_out = self.conv(state)
        conv_flat = conv_out.reshape(state.size(0), -1)

        # add previous move information to the convolution results
        combined = torch.cat([conv_flat, prev_move_onehot], dim=1)

        # forward pass
        q = self.fc(combined)
        
        return q
    
if __name__=="__main__":
    pass

