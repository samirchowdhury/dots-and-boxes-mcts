from dots_boxes_mcts.network import NetworkConfig, PolicyValueNetwork, torch


def test_network_config_matches_board_and_action_shapes() -> None:
    config = NetworkConfig(rows=3, cols=3)

    assert config.board_height == 5
    assert config.board_width == 5
    assert config.action_count == 12


def test_policy_value_network_forward_when_torch_is_available() -> None:
    if torch is None:
        return

    config = NetworkConfig(rows=3, cols=3, hidden_channels=8, residual_blocks=1)
    model = PolicyValueNetwork(config)
    x = torch.zeros((2, config.channels, config.board_height, config.board_width))
    legal_mask = torch.ones((2, config.action_count))

    policy_logits, values = model(x, legal_mask)

    assert policy_logits.shape == (2, config.action_count)
    assert values.shape == (2,)
