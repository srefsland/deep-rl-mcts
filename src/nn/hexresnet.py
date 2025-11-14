import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions."""

    def __init__(self, channels, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = activation()

    def forward(self, x):
        residual = x
        out = self.activation(self.conv1(x))
        out = self.conv2(out)

        out = out + residual
        out = self.activation(out)
        return out


class HexResNet(nn.Module):
    """AlphaGo Zero-style CNN for Hex (actor + critic)."""

    def __init__(self, board_size=6, in_channels=5, num_filters=64, num_res_blocks=3):
        super().__init__()
        self.board_size = board_size
        
        self.conv_in = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Actor head
        self.actor_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.actor_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Critic head
        self.critic_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.critic_fc1 = nn.Linear(board_size * board_size, 64)
        self.critic_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, channels, board_size, board_size)
        Returns:
            actor: Tensor (batch, board_size*board_size) - action probabilities
            critic: Tensor (batch, 1) - value estimates
        """
        x = self.activation(self.conv_in(x))
        x = self.res_blocks(x)

        # Actor head
        a = self.actor_conv(x)
        a = torch.flatten(a, start_dim=1)
        # return logits (no softmax) so callers can use nn.CrossEntropyLoss or other losses
        a = self.actor_fc(a)

        # Critic head
        c = self.critic_conv(x)
        c = torch.flatten(c, start_dim=1)
        c = F.relu(self.critic_fc1(c))
        c = torch.tanh(self.critic_fc2(c))

        return a, c

    def call_actor(self, x):
        """Inference with actor head output and softmax

        Args:
            x (_type_): _description_
        """
        in_ch = self.conv_in.in_channels

        # Accept single examples (3D) or batched inputs (4D). Detect whether
        # channels are first (C,H,W or N,C,H,W) or last (H,W,C or N,H,W,C).
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"call_actor expects a torch.Tensor, got {type(x)}")

        original_was_batched = x.dim() == 4

        if x.dim() == 3:
            # single example: (C,H,W) or (H,W,C)
            if x.shape[0] == in_ch:
                X = x.unsqueeze(0)  # (1,C,H,W)
            elif x.shape[-1] == in_ch:
                X = x.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
            else:
                raise ValueError(
                    f"call_actor input channels don't match model: expected {in_ch}, got shape={tuple(x.shape)}"
                )
        elif x.dim() == 4:
            # batched: (N,C,H,W) or (N,H,W,C)
            if x.shape[1] == in_ch:
                X = x
            elif x.shape[-1] == in_ch:
                X = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"call_actor input channels don't match model: expected {in_ch}, got shape={tuple(x.shape)}"
                )
        else:
            raise ValueError(f"call_actor expects a 3D or 4D tensor, got dim={x.dim()}")

        # Ensure float32 and contiguous
        if X.dtype != torch.float32:
            X = X.to(dtype=torch.float32)
        if not X.is_contiguous():
            X = X.contiguous()

        # Move to model device if needed
        device = next(self.parameters()).device
        if X.device != device:
            X = X.to(device)

        if getattr(X, "requires_grad", False):
            X = X.detach()

        # Inference
        self.eval()
        with torch.no_grad():
            logits, _ = self(X)
            probs = F.softmax(logits, dim=1)

        out = probs.cpu().numpy()
        if not original_was_batched:
            return out[0]
        return out

    def call_critic(self, x):
        """Inference with critic head output

        Args:
            x (_type_): _description_
        """
        in_ch = self.conv_in.in_channels

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"call_critic expects a torch.Tensor, got {type(x)}")

        original_was_batched = x.dim() == 4

        if x.dim() == 3:
            if x.shape[0] == in_ch:
                X = x.unsqueeze(0)
            elif x.shape[-1] == in_ch:
                X = x.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"call_critic input channels don't match model: expected {in_ch}, got shape={tuple(x.shape)}"
                )
        elif x.dim() == 4:
            if x.shape[1] == in_ch:
                X = x
            elif x.shape[-1] == in_ch:
                X = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(
                    f"call_critic input channels don't match model: expected {in_ch}, got shape={tuple(x.shape)}"
                )
        else:
            raise ValueError(f"call_critic expects a 3D or 4D tensor, got dim={x.dim()}")

        # Ensure float32 and contiguous
        if X.dtype != torch.float32:
            X = X.to(dtype=torch.float32)
        if not X.is_contiguous():
            X = X.contiguous()

        # Move to model device if needed
        device = next(self.parameters()).device
        if X.device != device:
            X = X.to(device)

        if getattr(X, "requires_grad", False):
            X = X.detach()

        # Inference
        self.eval()
        with torch.no_grad():
            _, values = self(X)

        out = values.cpu().numpy()
        # values shape is (N,1) -> squeeze last dim
        out = out.squeeze(-1)
        if not original_was_batched:
            return float(out[0])
        return out
