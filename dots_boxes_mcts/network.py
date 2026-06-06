from __future__ import annotations

from dataclasses import dataclass

from dots_boxes_mcts.encoding import CHANNEL_NAMES, action_ids, board_shape

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is absent.
    torch = None
    nn = None


@dataclass(frozen=True)
class NetworkConfig:
    rows: int = 3
    cols: int = 3
    channels: int = len(CHANNEL_NAMES)
    hidden_channels: int = 64
    residual_blocks: int = 4

    @property
    def action_count(self) -> int:
        return len(action_ids(self.rows, self.cols))

    @property
    def board_height(self) -> int:
        return board_shape(self.rows, self.cols)[0]

    @property
    def board_width(self) -> int:
        return board_shape(self.rows, self.cols)[1]


def require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for dots_boxes_mcts.network. "
            "Activate the data pyenv, then install torch when you are ready to train."
        )


if nn is not None:

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.activation = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.activation(x + self.layers(x))


    class PolicyValueNetwork(nn.Module):
        def __init__(self, config: NetworkConfig) -> None:
            super().__init__()
            self.config = config
            height = config.board_height
            width = config.board_width
            hidden = config.hidden_channels
            flat_size = height * width

            self.stem = nn.Sequential(
                nn.Conv2d(config.channels, hidden, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            )
            self.body = nn.Sequential(
                *[ResidualBlock(hidden) for _ in range(config.residual_blocks)]
            )
            self.policy_head = nn.Sequential(
                nn.Conv2d(hidden, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2 * flat_size, config.action_count),
            )
            self.value_head = nn.Sequential(
                nn.Conv2d(hidden, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(flat_size, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
                nn.Tanh(),
            )

        def forward(
            self,
            x: torch.Tensor,
            legal_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.body(self.stem(x))
            policy_logits = self.policy_head(features)
            if legal_mask is not None:
                policy_logits = policy_logits.masked_fill(legal_mask <= 0, -1.0e9)
            value = self.value_head(features).squeeze(-1)
            return policy_logits, value

else:

    class PolicyValueNetwork:  # type: ignore[no-redef]
        def __init__(self, config: NetworkConfig) -> None:
            require_torch()
