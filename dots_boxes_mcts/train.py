from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from dots_boxes_mcts.encoding import action_ids, action_index, encode_snapshot
from dots_boxes_mcts.self_play import write_jsonl


@dataclass(frozen=True)
class TrainingExample:
    state: dict
    player: int
    policy: dict[str, float]
    value: float
    selected_move: str
    source: dict


@dataclass(frozen=True)
class OverfitDiagnostics:
    epoch: int
    loss: float
    policy_loss: float
    policy_target_entropy: float
    policy_kl: float
    value_loss: float
    policy_top1_accuracy: float
    value_mae: float


@dataclass
class MlxPolicyValueNetwork:
    board_height: int
    board_width: int
    channels: int
    action_count: int
    hidden_size: int = 64
    residual_blocks: int = 4
    seed: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        mx = require_mlx(device=self.device)
        nn = require_mlx_nn(device=self.device)
        mx.random.seed(self.seed)
        self.module = ResidualPolicyValueModule(
            channels=self.channels,
            hidden_size=self.hidden_size,
            residual_blocks=self.residual_blocks,
            board_height=self.board_height,
            board_width=self.board_width,
            action_count=self.action_count,
            nn=nn,
        )

    def forward(self, x, legal_mask):
        return self.module(x, legal_mask)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            board_height=np.array([self.board_height]),
            board_width=np.array([self.board_width]),
            channels=np.array([self.channels]),
            action_count=np.array([self.action_count]),
            hidden_size=np.array([self.hidden_size]),
            residual_blocks=np.array([self.residual_blocks]),
            **{
                name: np.array(value)
                for name, value in flatten_parameter_tree(self.module.parameters()).items()
            },
        )


def load_mlx_checkpoint(path: Path, device: str = "cpu") -> MlxPolicyValueNetwork:
    data = np.load(path)
    model = MlxPolicyValueNetwork(
        board_height=int(data["board_height"][0]),
        board_width=int(data["board_width"][0]),
        channels=int(data["channels"][0]),
        action_count=int(data["action_count"][0]),
        hidden_size=int(data["hidden_size"][0]),
        residual_blocks=int(data["residual_blocks"][0]),
        device=device,
    )
    mx = require_mlx(device=device)
    arrays = {
        key: mx.array(data[key])
        for key in data.files
        if key
        not in {
            "board_height",
            "board_width",
            "channels",
            "action_count",
            "hidden_size",
            "residual_blocks",
        }
    }
    model.module.update(unflatten_parameter_tree(arrays))
    mx.eval(model.module.parameters())
    return model


class ResidualBlock:
    def __init__(self, channels: int, nn) -> None:
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(channels)
        self.nn = nn

    def __call__(self, x):
        return self.nn.relu(x + self.bn2(self.conv2(self.nn.relu(self.bn1(self.conv1(x))))))

    def parameters(self) -> dict:
        return {
            "conv1": self.conv1.parameters(),
            "bn1": self.bn1.parameters(),
            "conv2": self.conv2.parameters(),
            "bn2": self.bn2.parameters(),
        }


class ResidualPolicyValueModule:
    def __init__(
        self,
        channels: int,
        hidden_size: int,
        residual_blocks: int,
        board_height: int,
        board_width: int,
        action_count: int,
        nn,
    ) -> None:
        self.nn = nn
        self.stem_conv = nn.Conv2d(channels, hidden_size, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm(hidden_size)
        self.blocks = [ResidualBlock(hidden_size, nn=nn) for _ in range(residual_blocks)]
        self.policy_conv = nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm(2)
        self.policy_linear = nn.Linear(2 * board_height * board_width, action_count)
        self.value_conv = nn.Conv2d(hidden_size, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm(1)
        self.value_linear1 = nn.Linear(board_height * board_width, hidden_size)
        self.value_linear2 = nn.Linear(hidden_size, 1)

    def __call__(self, x, legal_mask):
        features = self.nn.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            features = block(features)
        policy_features = self.nn.relu(self.policy_bn(self.policy_conv(features)))
        policy_logits = self.policy_linear(flatten_spatial(policy_features))
        policy = mlx_masked_softmax(policy_logits, legal_mask)
        value_features = self.nn.relu(self.value_bn(self.value_conv(features)))
        value_hidden = self.nn.relu(self.value_linear1(flatten_spatial(value_features)))
        value = self.nn.tanh(self.value_linear2(value_hidden)).reshape((-1,))
        return policy, value

    def parameters(self) -> dict:
        return {
            "stem_conv": self.stem_conv.parameters(),
            "stem_bn": self.stem_bn.parameters(),
            "blocks": {
                str(index): block.parameters()
                for index, block in enumerate(self.blocks)
            },
            "policy_conv": self.policy_conv.parameters(),
            "policy_bn": self.policy_bn.parameters(),
            "policy_linear": self.policy_linear.parameters(),
            "value_conv": self.value_conv.parameters(),
            "value_bn": self.value_bn.parameters(),
            "value_linear1": self.value_linear1.parameters(),
            "value_linear2": self.value_linear2.parameters(),
        }

    def trainable_parameters(self) -> dict:
        return self.parameters()

    def update(self, parameters: dict) -> None:
        self.stem_conv.update(parameters["stem_conv"])
        self.stem_bn.update(parameters["stem_bn"])
        for index, block in enumerate(self.blocks):
            block.conv1.update(parameters["blocks"][str(index)]["conv1"])
            block.bn1.update(parameters["blocks"][str(index)]["bn1"])
            block.conv2.update(parameters["blocks"][str(index)]["conv2"])
            block.bn2.update(parameters["blocks"][str(index)]["bn2"])
        self.policy_conv.update(parameters["policy_conv"])
        self.policy_bn.update(parameters["policy_bn"])
        self.policy_linear.update(parameters["policy_linear"])
        self.value_conv.update(parameters["value_conv"])
        self.value_bn.update(parameters["value_bn"])
        self.value_linear1.update(parameters["value_linear1"])
        self.value_linear2.update(parameters["value_linear2"])


def require_mlx(device: str | None = None):
    try:
        import mlx.core as mx
    except ImportError as error:
        raise ImportError(
            "MLX is required for the Stage 3.1 overfit scaffold. "
            "Run `pyenv activate data && python -m pip install mlx`."
        ) from error

    if device is None:
        return mx
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    elif device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        raise ValueError("device must be 'cpu' or 'gpu'")
    return mx


def require_mlx_nn(device: str | None = None):
    require_mlx(device=device)
    try:
        import mlx.nn as nn
    except ImportError as error:
        raise ImportError(
            "MLX neural-network layers are required for the residual conv model."
        ) from error
    return nn


def mlx_runtime_available(device: str = "cpu") -> bool:
    try:
        mx = require_mlx(device=device)
        value = mx.array([1.0], dtype=mx.float32) + mx.array([1.0], dtype=mx.float32)
        mx.eval(value)
    except Exception:
        return False
    return True


def mlx_forward(params, x, legal_mask):
    raise RuntimeError("Use MlxPolicyValueNetwork.forward for the residual conv model.")


def mlx_masked_softmax(logits, legal_mask):
    mx = require_mlx()
    masked_logits = mx.where(legal_mask > 0, logits, mx.array(-1.0e9))
    shifted = masked_logits - mx.max(masked_logits, axis=1, keepdims=True)
    exp_logits = mx.exp(shifted) * legal_mask
    return exp_logits / mx.maximum(mx.sum(exp_logits, axis=1, keepdims=True), mx.array(1.0e-12))


def flatten_spatial(x):
    mx = require_mlx()
    return mx.flatten(x, start_axis=1)


def flatten_parameter_tree(tree: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_parameter_tree(value, prefix=name))
        else:
            flat[name] = value
    return flat


def unflatten_parameter_tree(flat: dict[str, object]) -> dict:
    tree: dict = {}
    for name, value in flat.items():
        cursor = tree
        parts = name.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return tree


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def examples_from_records(records: Iterable[dict]) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for record_index, record in enumerate(records):
        for decision_index, decision in enumerate(record.get("decisions", [])):
            examples.append(
                example_from_decision(
                    record=record,
                    decision=decision,
                    record_index=record_index,
                    decision_index=decision_index,
                )
            )
    return examples


def examples_from_payloads(payloads: Iterable[dict]) -> list[TrainingExample]:
    payloads = list(payloads)
    if not payloads:
        return []
    if all("decisions" in payload for payload in payloads):
        return examples_from_records(payloads)
    if all({"state", "player", "policy", "value"}.issubset(payload) for payload in payloads):
        return [example_from_payload(payload) for payload in payloads]
    raise ValueError("Inputs must be all game records or all serialized training examples.")


def example_from_payload(payload: dict) -> TrainingExample:
    return TrainingExample(
        state=payload["state"],
        player=int(payload["player"]),
        policy={move: float(probability) for move, probability in payload["policy"].items()},
        value=float(payload["value"]),
        selected_move=payload.get("selectedMove", ""),
        source=payload.get("source", {}),
    )


def example_from_decision(
    record: dict,
    decision: dict,
    record_index: int = 0,
    decision_index: int = 0,
) -> TrainingExample:
    state = decision["state"]
    player = int(decision.get("player", state["currentPlayer"]))
    return TrainingExample(
        state=state,
        player=player,
        policy=policy_from_search(state, decision["search"]),
        value=value_from_record(record, player),
        selected_move=decision["search"]["move"],
        source={
            "recordIndex": record_index,
            "decisionIndex": decision_index,
            "turn": decision.get("turn"),
            "seed": record.get("seed"),
        },
    )


def policy_from_search(state: dict, search: dict) -> dict[str, float]:
    visits_by_move = {stat["move"]: int(stat["visits"]) for stat in search.get("stats", [])}
    total_visits = sum(visits_by_move.values())
    if total_visits <= 0:
        move = search["move"]
        return {move: 1.0}

    policy: dict[str, float] = {}
    for move in action_ids(int(state["rows"]), int(state["cols"])):
        visits = visits_by_move.get(move, 0)
        if visits > 0:
            policy[move] = visits / total_visits
    return policy


def value_from_record(record: dict, player: int) -> float:
    scores = record["finalScores"]
    opponent = 1 if player == 0 else 0
    total_boxes = max((int(record["rows"]) - 1) * (int(record["cols"]) - 1), 1)
    return (float(scores[player]) - float(scores[opponent])) / total_boxes


def policy_vector(example: TrainingExample) -> np.ndarray:
    rows = int(example.state["rows"])
    cols = int(example.state["cols"])
    vector = np.zeros(len(action_ids(rows, cols)), dtype=np.float32)
    for move, probability in example.policy.items():
        vector[action_index(move, rows, cols)] = probability
    return vector


def serializable_example(example: TrainingExample, include_encoding_summary: bool = False) -> dict:
    payload = {
        "state": example.state,
        "player": example.player,
        "policy": example.policy,
        "value": example.value,
        "selectedMove": example.selected_move,
        "source": example.source,
    }
    if include_encoding_summary:
        encoded = encode_snapshot(example.state)
        payload["encoding"] = {
            "tensorShape": list(encoded.tensor.shape),
            "legalMoves": int(encoded.legal_mask.sum()),
            "drawnEdges": int(encoded.tensor[0].sum()),
            "currentPlayerEdges": int(encoded.tensor[1].sum()),
            "opponentEdges": int(encoded.tensor[2].sum()),
            "currentPlayerBoxes": int(encoded.tensor[3].sum()),
            "opponentBoxes": int(encoded.tensor[4].sum()),
            "policyVectorNonZero": int(np.count_nonzero(policy_vector(example))),
        }
    return payload


def write_examples(examples: list[TrainingExample], out_path: Path) -> None:
    write_jsonl([serializable_example(example) for example in examples], out_path)


def tensors_from_examples(
    examples: list[TrainingExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("Need at least one training example.")

    encoded_positions = [encode_snapshot(example.state) for example in examples]
    x = np.stack([np.moveaxis(position.tensor, 0, -1) for position in encoded_positions]).astype(
        np.float32
    )
    policy_target = np.stack([policy_vector(example) for example in examples]).astype(np.float32)
    value_target = np.array([example.value for example in examples], dtype=np.float32)
    legal_mask = np.stack([position.legal_mask for position in encoded_positions]).astype(np.float32)
    return x, policy_target, value_target, legal_mask


def overfit_diagnostics(
    epoch: int,
    policy: np.ndarray,
    value: np.ndarray,
    policy_target: np.ndarray,
    value_target: np.ndarray,
    value_weight: float = 1.0,
) -> OverfitDiagnostics:
    policy_loss = float((-policy_target * np.log(np.maximum(policy, 1.0e-12))).sum(axis=1).mean())
    policy_target_entropy = float(
        (-policy_target * np.log(np.maximum(policy_target, 1.0e-12))).sum(axis=1).mean()
    )
    value_loss = float(((value - value_target) ** 2).mean())
    return OverfitDiagnostics(
        epoch=epoch,
        loss=policy_loss + value_weight * value_loss,
        policy_loss=policy_loss,
        policy_target_entropy=policy_target_entropy,
        policy_kl=policy_loss - policy_target_entropy,
        value_loss=value_loss,
        policy_top1_accuracy=float((policy.argmax(axis=1) == policy_target.argmax(axis=1)).mean()),
        value_mae=float(np.abs(value - value_target).mean()),
    )


def overfit_examples(
    examples: list[TrainingExample],
    epochs: int = 300,
    learning_rate: float = 0.1,
    hidden_size: int = 64,
    seed: int = 1,
    value_weight: float = 1.0,
    diagnostics_every: int = 50,
    device: str = "cpu",
    residual_blocks: int = 4,
) -> tuple[MlxPolicyValueNetwork, list[OverfitDiagnostics]]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    x, policy_target, value_target, legal_mask = tensors_from_examples(examples)
    mx = require_mlx(device=device)
    mlx_x = mx.array(x)
    mlx_policy_target = mx.array(policy_target)
    mlx_value_target = mx.array(value_target)
    mlx_legal_mask = mx.array(legal_mask)
    model = MlxPolicyValueNetwork(
        board_height=x.shape[1],
        board_width=x.shape[2],
        channels=x.shape[3],
        action_count=policy_target.shape[1],
        hidden_size=hidden_size,
        residual_blocks=residual_blocks,
        seed=seed,
        device=device,
    )
    optimizer = mlx_optimizer(learning_rate=learning_rate, device=device)
    diagnostics: list[OverfitDiagnostics] = []
    for epoch in range(1, epochs + 1):
        train_model_step(
            model=model,
            optimizer=optimizer,
            x=mlx_x,
            policy_target=mlx_policy_target,
            value_target=mlx_value_target,
            legal_mask=mlx_legal_mask,
            value_weight=value_weight,
        )
        if epoch == 1 or epoch == epochs or epoch % diagnostics_every == 0:
            policy, value = model.forward(mlx_x, mlx_legal_mask)
            mx.eval(policy, value)
            diagnostics.append(
                overfit_diagnostics(
                    epoch=epoch,
                    policy=np.array(policy),
                    value=np.array(value),
                    policy_target=policy_target,
                    value_target=value_target,
                    value_weight=value_weight,
                )
            )
    return model, diagnostics


def train_checkpoint(
    examples: list[TrainingExample],
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.05,
    hidden_size: int = 128,
    seed: int = 1,
    value_weight: float = 1.0,
    validation_fraction: float = 0.1,
    diagnostics_every: int = 1,
    device: str = "cpu",
    residual_blocks: int = 4,
) -> tuple[MlxPolicyValueNetwork, list[dict]]:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    x, policy_target, value_target, legal_mask = tensors_from_examples(examples)
    train_indices, validation_indices = train_validation_indices(
        example_count=len(examples),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    mx = require_mlx(device=device)
    rng = np.random.default_rng(seed)
    model = MlxPolicyValueNetwork(
        board_height=x.shape[1],
        board_width=x.shape[2],
        channels=x.shape[3],
        action_count=policy_target.shape[1],
        hidden_size=hidden_size,
        residual_blocks=residual_blocks,
        seed=seed,
        device=device,
    )
    optimizer = mlx_optimizer(learning_rate=learning_rate, device=device)
    diagnostics: list[dict] = []

    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            train_model_step(
                model=model,
                optimizer=optimizer,
                x=mx.array(x[batch]),
                policy_target=mx.array(policy_target[batch]),
                value_target=mx.array(value_target[batch]),
                legal_mask=mx.array(legal_mask[batch]),
                value_weight=value_weight,
            )

        if epoch == 1 or epoch == epochs or epoch % diagnostics_every == 0:
            diagnostics.append(
                {
                    "epoch": epoch,
                    "split": "train",
                    **serializable_diagnostics(
                        diagnostics_for_arrays(
                            model=model,
                            x=x[train_indices],
                            policy_target=policy_target[train_indices],
                            value_target=value_target[train_indices],
                            legal_mask=legal_mask[train_indices],
                            value_weight=value_weight,
                            epoch=epoch,
                        )
                    ),
                }
            )
            if len(validation_indices) > 0:
                diagnostics.append(
                    {
                        "epoch": epoch,
                        "split": "validation",
                        **serializable_diagnostics(
                            diagnostics_for_arrays(
                                model=model,
                                x=x[validation_indices],
                                policy_target=policy_target[validation_indices],
                                value_target=value_target[validation_indices],
                                legal_mask=legal_mask[validation_indices],
                                value_weight=value_weight,
                                epoch=epoch,
                            )
                        ),
                    }
                )

    return model, diagnostics


def mlx_optimizer(learning_rate: float, device: str):
    require_mlx(device=device)
    try:
        import mlx.optimizers as optim
    except ImportError as error:
        raise ImportError("MLX optimizers are required for training.") from error
    return optim.Adam(learning_rate=learning_rate)


def train_model_step(
    model: MlxPolicyValueNetwork,
    optimizer,
    x,
    policy_target,
    value_target,
    legal_mask,
    value_weight: float,
) -> None:
    mx = require_mlx(device=model.device)
    nn = require_mlx_nn(device=model.device)

    def loss_fn(module, batch_x, batch_policy_target, batch_value_target, batch_legal_mask):
        policy, value = module(batch_x, batch_legal_mask)
        policy_loss = mx.mean(
            -mx.sum(
                batch_policy_target * mx.log(mx.maximum(policy, mx.array(1.0e-12))),
                axis=1,
            )
        )
        value_loss = mx.mean((value - batch_value_target) ** 2)
        return policy_loss + value_weight * value_loss

    _, grads = nn.value_and_grad(model.module, loss_fn)(
        model.module,
        x,
        policy_target,
        value_target,
        legal_mask,
    )
    model.module.update(optimizer.apply_gradients(grads, model.module.parameters()))
    mx.eval(model.module.parameters(), optimizer.state)


def train_validation_indices(
    example_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(example_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    validation_count = int(round(example_count * validation_fraction))
    if validation_fraction > 0.0:
        validation_count = max(1, validation_count)
    validation_indices = np.sort(indices[:validation_count])
    train_indices = np.sort(indices[validation_count:])
    if len(train_indices) == 0:
        raise ValueError("Need at least one training example after validation split.")
    return train_indices, validation_indices


def diagnostics_for_arrays(
    model: MlxPolicyValueNetwork,
    x: np.ndarray,
    policy_target: np.ndarray,
    value_target: np.ndarray,
    legal_mask: np.ndarray,
    value_weight: float,
    epoch: int,
) -> OverfitDiagnostics:
    mx = require_mlx(device=model.device)
    policy, value = model.forward(mx.array(x), mx.array(legal_mask))
    mx.eval(policy, value)
    return overfit_diagnostics(
        epoch=epoch,
        policy=np.array(policy),
        value=np.array(value),
        policy_target=policy_target,
        value_target=value_target,
        value_weight=value_weight,
    )


def serializable_diagnostics(diagnostics: OverfitDiagnostics) -> dict:
    return {
        "epoch": diagnostics.epoch,
        "loss": diagnostics.loss,
        "policyLoss": diagnostics.policy_loss,
        "policyTargetEntropy": diagnostics.policy_target_entropy,
        "policyKl": diagnostics.policy_kl,
        "valueLoss": diagnostics.value_loss,
        "policyTop1Accuracy": diagnostics.policy_top1_accuracy,
        "valueMae": diagnostics.value_mae,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AlphaZero-style examples from MCTS JSONL.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overfit-epochs", type=int, default=0)
    parser.add_argument("--train-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--diagnostics-every", type=int, default=50)
    parser.add_argument("--diagnostics-out", type=Path)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--mlx-device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    payloads: list[dict] = []
    for path in args.inputs:
        payloads.extend(load_jsonl(path))

    examples = examples_from_payloads(payloads)
    if args.limit > 0:
        examples = examples[: args.limit]

    if args.out is not None:
        write_examples(examples, args.out)

    if args.preview or (args.out is None and args.overfit_epochs <= 0 and args.train_epochs <= 0):
        for example in examples:
            print(json.dumps(serializable_example(example, include_encoding_summary=True), sort_keys=True))

    if args.overfit_epochs > 0:
        model, diagnostics = overfit_examples(
            examples=examples,
            epochs=args.overfit_epochs,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            residual_blocks=args.residual_blocks,
            value_weight=args.value_weight,
            diagnostics_every=args.diagnostics_every,
            device=args.mlx_device,
        )
        for item in diagnostics:
            print(json.dumps(serializable_diagnostics(item), sort_keys=True))
        if args.checkpoint_out is not None:
            model.save(args.checkpoint_out)
            print(f"Wrote tiny overfit checkpoint to {args.checkpoint_out}")

    if args.train_epochs > 0:
        model, diagnostics = train_checkpoint(
            examples=examples,
            epochs=args.train_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            residual_blocks=args.residual_blocks,
            value_weight=args.value_weight,
            validation_fraction=args.validation_fraction,
            diagnostics_every=args.diagnostics_every,
            device=args.mlx_device,
        )
        for item in diagnostics:
            print(json.dumps(item, sort_keys=True))
        if args.diagnostics_out is not None:
            write_jsonl(diagnostics, args.diagnostics_out)
            print(f"Wrote training diagnostics to {args.diagnostics_out}")
        if args.checkpoint_out is not None:
            model.save(args.checkpoint_out)
            print(f"Wrote training checkpoint to {args.checkpoint_out}")

    if args.out is not None:
        print(f"Wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
