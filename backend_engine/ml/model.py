import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    defines the residual block for our neural net follows the architecture used by AlphaZero
    the residual block has two 3x3 convolutional layers with skip connections to prevetn vanishing gradients

    Args:
        nn (nn.Module): PyTorch neural network module
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        # pad 8x8 board to 10x10 with padding=1 so that the 3x3 conv doesn't reduce size
        # batch normilization stabilizes training and means we dont need bias in conv layers
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, input_tensor):
        """
        performs a forward pass through the residual block

        Args:
            input_tensor (torch.Tensor): input tensor

        Returns:
            out (torch.Tensor): output tensor
        """
        # store input for skip connection
        residual = input_tensor

        # first layer
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = F.relu(x)

        # second layer
        x = self.conv2(x)
        x = self.bn2(x)

        # add skip connection
        x += residual

        # relu activation
        output_tensor = F.relu(x)
        return output_tensor


class ChessValueNet(nn.Module):
    """
    Main Value Net Powering the chess engine
    the net takes the board state represented as a tensor
    and returns a scalar value between -1 and 1 indicating
    "probability of a white win" or the strength of the position
    """

    def __init__(self, num_channels=64, num_residual_blocks=5):
        super(ChessValueNet, self).__init__()

        # first pass the input tensor through a conv layer to deepen the representation
        self.conv = nn.Conv2d(
            in_channels=20,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
            bias=False
            )
        self.bn = nn.BatchNorm2d(num_channels)

        # create residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        )

        # then flatten and pass through a linear layer to get a single scalar output
        # pass it through a 1x1 conv to flatten channels to 1
        self.conv_final = nn.Conv2d(
            in_channels=num_channels,
            out_channels=1,
            kernel_size=1,
            bias=False
        )
        # conv_final: (batch_size, 1, 8, 8)
        self.bn_final = nn.BatchNorm2d(1)

        # and pass to two fully conneced linear layers to get final prediction
        # 1 x 8 x 8  ->  64
        # the expand to 128 neruons for a better representation
        # then finally squash back down to a scalar output
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)

        # init weights
        self._init_weights()

    def _init_weights(self):
        """
        initializes the weights of the network using Kaiming He initialization
        for convolutional layers
        Xavier initialization for linear layers
        and constant initialization for batch norm layers
        """
        for model in self.modules():
            if isinstance(model, nn.Conv2d):
                # Kaiming He initialization for conv layers
                nn.init.kaiming_normal_(
                    model.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                    )
            elif isinstance(model, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_normal_(model.weight)
                if model.bias is not None:
                    nn.init.constant_(model.bias, 0)
            elif isinstance(model, nn.BatchNorm2d):
                # Constant initialization for batch norm layers
                nn.init.constant_(model.weight, 1)
                nn.init.constant_(model.bias, 0)

    def forward(self, board_tensors):
        """
        performs the forward pass through the network
        Args:
            board_tensors (torch.Tensor): input tensor of shape (batch_size, 20, 8, 8)
        Returns:
            win_probs (torch.Tensor): output tensor of shape (batch_size, 1)
        """
        # initial conv layer to build representation space
        # (batch_size, 20, 8, 8) -> (batch_size, num_channels, 8, 8)
        x = self.conv(board_tensors)
        x = self.bn(x)
        x = F.relu(x)

        # pass through residual blocks to get deep features
        # (batch_size, num_channels, 8, 8) -> (batch_size, num_channels, 8, 8)
        for block in self.residual_blocks:
            x = block(x)

        # final conv layer to reduce channels to 1 to extract a single value map
        # (batch_size, num_channels, 8, 8) -> (batch_size, 1, 8, 8)
        x = self.conv_final(x)
        x = self.bn_final(x)
        x = F.relu(x)

        # flatten the 1x8x8 to 64
        # (batch_size, 1, 8, 8) -> (batch_size, 64)
        x = x.view(x.size(0), -1)

        # pass through fully connected layers to get final scalar output
        # (batch_size, 64) -> (batch_size, 128) -> (batch_size, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # scale output to win probability between -1 and 1 using tanh
        win_probs = torch.tanh(x)

        return win_probs
