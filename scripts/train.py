from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from omegaconf import OmegaConf

from mave_repro.core.utils import (
    ensure_dir,
    set_seed,
)

from mave_repro.mave.curve_fit import (
    CurveFitConfig,
)

from mave_repro.mave.rollout_group import (
    MAVEGroupConfig,
    MAVEGroupProcessor,
)

from mave_repro.training.alternating import (
    AlternatingTrainConfig,
    AlternatingTrainer,
    TrainingPhase,
)

from mave_repro.training.train_escalation import (
    EscalationTrainConfig,
    EscalationTrainer,
)

from mave_repro.training.train_reaction import (
    ReactionTrainConfig,
    ReactionTrainer,
)


# ============================================================
# Config
# ============================================================


def load_yaml(
    path: Path,
) -> dict[str, Any]:

    config = OmegaConf.load(
        path
    )

    return OmegaConf.to_container(
        config,
        resolve=True,
    )


def build_config_bundle(
    config_dir: Path,
    *,
    oracle_name: str = "hierarchy",
) -> dict[str, Any]:

    return {
        "base":
            load_yaml(
                config_dir
                / "base.yaml"
            ),

        "model":
            load_yaml(
                config_dir
                / "model"
                / "llama3_8b.yaml"
            ),

        "planner":
            load_yaml(
                config_dir
                / "planner"
                / "mcts.yaml"
            ),

        "oracle":
            load_yaml(
                config_dir
                / "oracle"
                / f"{oracle_name}.yaml"
            ),

        "mave_train":
            load_yaml(
                config_dir
                / "train"
                / "mave.yaml"
            ),

        "reaction_train":
            load_yaml(
                config_dir
                / "train"
                / "reaction_policy.yaml"
            ),
    }


# ============================================================
# Runtime-factory hook
# ============================================================


def load_symbol(
    spec: str,
) -> Callable[..., Any]:
    """
    Load:

        package.module:function
    """

    if ":" not in spec:
        raise ValueError(
            "Factory must use "
            "'module:function' format."
        )

    module_name, symbol_name = (
        spec.split(
            ":",
            maxsplit=1,
        )
    )

    module = (
        importlib.import_module(
            module_name
        )
    )

    symbol = getattr(
        module,
        symbol_name,
    )

    if not callable(symbol):
        raise TypeError(
            f"{spec!r} is not callable."
        )

    return symbol


# ============================================================
# Trainer construction
# ============================================================


def build_trainers(
    config: dict[str, Any],
    runtime: dict[str, Any],
) -> AlternatingTrainer:

    mave_cfg = (
        config["mave_train"]
    )

    reaction_cfg = (
        config["reaction_train"]
    )

    # --------------------------------------------------------
    # MAVE processor
    # --------------------------------------------------------

    curve_cfg = CurveFitConfig(
        mapping=(
            mave_cfg[
                "reward_cost_mapping"
            ]["type"]
        ),

        eta=float(
            mave_cfg[
                "reward_cost_mapping"
            ]["regularization"][
                "eta"
            ]
        ),
    )

    group_cfg = MAVEGroupConfig(
        expected_group_size=int(
            mave_cfg[
                "rollout"
            ]["group_size"]
        ),

        lambda_cost=float(
            mave_cfg[
                "objective"
            ]["lambda_cost"]
        ),

        beta1=float(
            mave_cfg[
                "advantage"
            ]["components"][
                "D1"
            ]["weight"]
        ),

        beta2=float(
            mave_cfg[
                "advantage"
            ]["components"][
                "D2"
            ]["weight"]
        ),

        normalization_eps=float(
            mave_cfg[
                "advantage"
            ]["epsilon"]
        ),

        curve_fit=curve_cfg,
    )

    mave_processor = (
        MAVEGroupProcessor(
            group_cfg
        )
    )

    # --------------------------------------------------------
    # Escalation trainer
    # --------------------------------------------------------

    escalation_cfg = (
        EscalationTrainConfig(
            learning_rate=float(
                mave_cfg[
                    "grpo"
                ]["learning_rate"]
            ),

            clip_epsilon=float(
                mave_cfg[
                    "grpo"
                ]["clipping_range"]
            ),

            kl_coefficient=float(
                mave_cfg[
                    "grpo"
                ]["kl"][
                    "coefficient"
                ]
            ),

            max_grad_norm=(
                mave_cfg[
                    "optimization"
                ].get(
                    "max_grad_norm"
                )
            ),
        )
    )

    escalation_trainer = (
        EscalationTrainer(
            policy=runtime[
                "escalation_policy"
            ],

            reference_policy=runtime[
                "reference_escalation_policy"
            ],

            mave_processor=(
                mave_processor
            ),

            config=(
                escalation_cfg
            ),

            optimizer=runtime.get(
                "escalation_optimizer"
            ),

            trainable_parameters=(
                runtime.get(
                    "escalation_trainable_parameters"
                )
            ),
        )
    )

    # --------------------------------------------------------
    # Reaction trainer
    # --------------------------------------------------------

    reaction_grpo = (
        reaction_cfg["grpo"]
    )

    reaction_optimization = (
        reaction_cfg[
            "optimization"
        ]
    )

    reaction_train_config = (
        ReactionTrainConfig(
            learning_rate=float(
                reaction_grpo[
                    "learning_rate"
                ]
            ),

            clip_epsilon=float(
                reaction_grpo[
                    "clipping_range"
                ]
            ),

            kl_coefficient=float(
                reaction_grpo[
                    "kl"][
                    "coefficient"
                ]
            ),

            max_grad_norm=(
                reaction_optimization.get(
                    "max_grad_norm"
                )
            ),
        )
    )

    reaction_trainer = (
        ReactionTrainer(
            policy=runtime[
                "reaction_policy"
            ],

            reference_policy=runtime[
                "reference_reaction_policy"
            ],

            config=(
                reaction_train_config
            ),

            optimizer=runtime.get(
                "reaction_optimizer"
            ),

            trainable_parameters=(
                runtime.get(
                    "reaction_trainable_parameters"
                )
            ),
        )
    )

    # --------------------------------------------------------
    # Alternation
    # --------------------------------------------------------

    interval = int(
        mave_cfg[
            "alternating"
        ]["interval"]
    )

    epochs = int(
        mave_cfg[
            "optimization"
        ]["epochs"]
    )

    batch_size = int(
        mave_cfg[
            "optimization"
        ]["batch_size"]
    )

    alternating_cfg = (
        AlternatingTrainConfig(
            epochs=epochs,
            alternating_interval=(
                interval
            ),
            batch_size=(
                batch_size
            ),
            start_phase=(
                TrainingPhase
                .ESCALATION
            ),
        )
    )

    return AlternatingTrainer(
        escalation_trainer=(
            escalation_trainer
        ),

        reaction_trainer=(
            reaction_trainer
        ),

        escalation_batch_provider=(
            runtime[
                "escalation_batch_provider"
            ]
        ),

        reaction_batch_provider=(
            runtime[
                "reaction_batch_provider"
            ]
        ),

        config=alternating_cfg,
    )


# ============================================================
# Logging / checkpoint
# ============================================================


def save_runtime_checkpoint(
    runtime: dict[str, Any],
    path: Path,
) -> None:

    checkpointables = (
        runtime.get(
            "checkpointables",
            {},
        )
    )

    if not checkpointables:
        return

    import torch

    state = {}

    for name, obj in (
        checkpointables.items()
    ):

        if hasattr(
            obj,
            "state_dict",
        ):
            state[name] = (
                obj.state_dict()
            )

        elif hasattr(
            obj,
            "model",
        ) and hasattr(
            obj.model,
            "state_dict",
        ):
            state[name] = (
                obj.model.state_dict()
            )

        else:
            raise TypeError(
                f"Checkpointable {name!r} "
                "has no state_dict()."
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        state,
        path,
    )


# ============================================================
# Main
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(
            "configs"
        ),
    )

    parser.add_argument(
        "--oracle",
        choices=[
            "hierarchy",
            "yield_only",
        ],
        default="hierarchy",
    )

    parser.add_argument(
        "--runtime-factory",
        type=str,
        required=True,
        help=(
            "Dotted runtime builder in "
            "'module:function' format."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/train"
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    config = build_config_bundle(
        args.config_dir,
        oracle_name=args.oracle,
    )

    seed = int(
        config["base"].get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    output_dir = ensure_dir(
        args.output_dir
    )

    # --------------------------------------------------------
    # Save exact configuration used.
    # --------------------------------------------------------

    with (
        output_dir
        / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # --------------------------------------------------------
    # Runtime components not uniquely reconstructable from paper.
    # --------------------------------------------------------

    factory = load_symbol(
        args.runtime_factory
    )

    runtime = factory(
        config
    )

    required = {
        "escalation_policy",
        "reaction_policy",
        "reference_escalation_policy",
        "reference_reaction_policy",
        "escalation_batch_provider",
        "reaction_batch_provider",
    }

    missing = (
        required
        - set(runtime)
    )

    if missing:
        raise KeyError(
            "Runtime factory missing keys: "
            + ", ".join(
                sorted(missing)
            )
        )

    trainer = build_trainers(
        config,
        runtime,
    )

    log_path = (
        output_dir
        / "training.jsonl"
    )

    def callback(record) -> None:

        payload = {
            "epoch":
                record.epoch,

            "phase":
                record.phase.value,
        }

        stats = (
            record.escalation_stats
            if record.escalation_stats
            is not None
            else record.reaction_stats
        )

        if stats is not None:
            payload.update(
                stats.__dict__
            )

        with log_path.open(
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

        if (
            args.checkpoint_every > 0
            and (
                record.epoch + 1
            )
            % args.checkpoint_every
            == 0
        ):
            save_runtime_checkpoint(
                runtime,
                output_dir
                / "checkpoints"
                / (
                    f"epoch_"
                    f"{record.epoch + 1:05d}"
                    f".pt"
                ),
            )

        print(
            f"[epoch "
            f"{record.epoch + 1}] "
            f"{record.phase.value}"
        )

    trainer.train(
        callback=callback
    )

    save_runtime_checkpoint(
        runtime,
        output_dir
        / "checkpoints"
        / "final.pt",
    )


if __name__ == "__main__":
    main()