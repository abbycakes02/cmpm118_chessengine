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


class ChessNet(nn.Module):
    """
    Main Value Net Powering the chess engine
    the net takes the board state represented as a tensor
    and returns a scalar value between -1 and 1 indicating
    "probability of a white win" or the strength of the position
    """

    def __init__(self, vocab_size, history_length=5, board_channels=12, hidden_channels=64, num_blocks=5):
        """
        initializes the Chess Value Network

        Args:
            history_length (int): number of past board states to include in the input
            in_channels (int): number of input channels per board state
            hidden_channels (int): number of channels in the hidden layers
            num_blocks (int): number of residual blocks in the network
        """
        super(ChessNet, self).__init__()

        # input size scales with history length
        self.input_channels = history_length * board_channels

        # first pass the input tensor through a conv layer
        # extracts features from the input tensor and maps to
        # a higher dimensional representation space
        self.conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False
            )
        self.bn = nn.BatchNorm2d(hidden_channels)

        # create residual blocks
        # residual blocks act as the brain, extracting deep features
        # from the board representation and learning complex patterns
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )

        # take the output of the residual blocks and pass it through the policy head
        # similar to alpha zero, 2 channels keeps some spatial structure
        self.policy_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=2,
            kernel_size=1,
            padding=1,
            bias=False
        )
        self.policy_bn = nn.BatchNorm2d(2)
        # policy head fully connected layer to output move probabilities
        fc_size = 2 * 8 * 8  # 2 channels, 8x8 board
        self.policy_fc = nn.Linear(fc_size, vocab_size)

        # then pass through the value head to get win probability
        # final conv layer to reduce channels to 1 to extract a single value map
        self.value_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            bias=False
        )
        # conv_final: (batch_size, 1, 8, 8)
        self.value_bn = nn.BatchNorm2d(1)

        # and pass to two fully conneced linear layers to get final prediction
        # by squishing the represenation down to 64 neurons before prediction
        # the model is forced to learn a smart representation as opposed to memorizing
        # the expand to 256 neurons to allow the model a better representation for score
        # then finally squash back down to a scalar output
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)

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

        # policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        # flatten for fully connected layer
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # final conv layer to reduce channels to 1 to extract a single value map
        # (batch_size, num_channels, 8, 8) -> (batch_size, 1, 8, 8)
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)

        # flatten the 1x8x8 to 64
        # (batch_size, 1, 8, 8) -> (batch_size, 64)
        v = v.view(v.size(0), -1)

        # pass through fully connected layers to get final scalar output
        # (batch_size, 64) -> (batch_size, 128) -> (batch_size, 1)
        v = self.fc1(v)
        v = F.relu(v)
        value = self.fc2(v)
        # scale output to win probability between -1 and 1 using tanh
        win_probs = torch.tanh(value)

        return policy_logits, win_probs
