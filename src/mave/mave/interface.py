from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from mave.environment.purchasable import (
    InMemoryPurchasableDatabase,
    PurchasableDatabase,
)
from mave.environment.reward import (
    RewardFunction,
    SparseTerminalReward,
    ZeroReward,
)
from mave.environment.retro_env import RetroEnvironment

from mave.models.backbone import LLMBackbone
from mave.models.escalation_policy import EscalationPolicy
from mave.models.reaction_policy import ReactionPolicy
from mave.models.single_step import (
    SingleStepModel,
    TemplateBasedSingleStepModel,
    TemplatePredictorBackend,
)

from mave.oracle.backends.depth_traversal import (
    DepthLimitedTraversalBackend,
    DepthTraversalBackend,
    TraversalConfig,
)
from mave.oracle.backends.round_trip import RoundTripBackend
from mave.oracle.backends.yield_model import YieldModelBackend
from mave.oracle.base import MultiFidelityOracleSystem
from mave.oracle.cache import (
    InMemoryFeedbackCache,
    NullFeedbackCache,
)
from mave.oracle.comparative import ComparativeOracle
from mave.oracle.cost import AcquisitionCostModel
from mave.oracle.evaluative import EvaluativeOracle
from mave.oracle.query_factory import (
    FeedbackQueryFactory,
    FeedbackQueryFactoryConfig,
)
from mave.oracle.strategic import StrategicOracle
from mave.oracle.structural import StructuralOracle

from mave.planning.mcts import MCTSConfig, MCTSPlanner
from mave.planning.rollout import (
    PlanningRolloutConfig,
    PlanningRolloutRunner,
)

from mave.training.alternating import (
    EscalationBatchProvider,
    ReactionBatchProvider,
)
from mave.training.collector import (
    EscalationGroupCollector,
    ReactionGroupCollector,
)
from mave.training.parameter_isolation import ensure_disjoint_parameters


# ============================================================
# Errors
# ============================================================


class SystemPreparationError(RuntimeError):
    """
    Raised when the repository cannot construct the requested system from
    the supplied configuration and concrete model providers.

    This is intentionally fail-fast: missing experiment assets should never
    be replaced by silent defaults.
    """


# ============================================================
# External experiment providers
# ============================================================


@dataclass(slots=True)
class SystemProviders:
    """
    Concrete experiment-specific components supplied to the official MAVE
    implementation through explicit in-process placeholders.

    The MAVE repository defines stable interfaces for these components.
    Datasets, licensed assets, and pretrained checkpoints for the Chen-style
    single-step model, the round-trip model, and the yield model are connected
    here by the experiment provider module.

    Fields may remain ``None`` while experiment assets are unavailable. The
    system assembly remains in prepare_system(), which raises a focused error
    when execution reaches a component whose concrete asset is still absent.
    """

    # --------------------------------------------------------
    # Required external surrogate/model implementations
    # --------------------------------------------------------

    single_step_backend: TemplatePredictorBackend | None = None
    round_trip_backend: RoundTripBackend | None = None
    yield_backend: YieldModelBackend | None = None

    # A complete single-step model may be supplied instead of its backend.
    single_step_model: SingleStepModel | None = None

    # --------------------------------------------------------
    # Optional preconstructed policy/reference LLMs
    # --------------------------------------------------------

    escalation_backbone: LLMBackbone | None = None
    reaction_backbone: LLMBackbone | None = None

    # Deprecated shared trainable backbone. It remains as an explicit error
    # path so existing provider modules fail with a migration message instead
    # of silently violating the alternating-optimization contract.
    backbone: LLMBackbone | None = None

    # The two reference policies may share this backbone because it is frozen.
    reference_backbone: LLMBackbone | None = None

    # --------------------------------------------------------
    # Optional environment overrides
    # --------------------------------------------------------

    purchasable_db: PurchasableDatabase | None = None
    reward_fn: RewardFunction | None = None

    # L4 can use the repository's depth-limited traversal by default.
    depth_traversal_backend: DepthTraversalBackend | None = None

    # --------------------------------------------------------
    # Training-only providers
    # --------------------------------------------------------

    escalation_batch_provider: EscalationBatchProvider | None = None
    reaction_batch_provider: ReactionBatchProvider | None = None

    # --------------------------------------------------------
    # Evaluation-only optional components
    # --------------------------------------------------------

    route_quality_evaluator: Any | None = None
    difficulty_labels: Mapping[str, str] | None = None

    # Extra objects that should be checkpointed by scripts/train.py.
    checkpointables: Mapping[str, Any] = field(default_factory=dict)


# ============================================================
# Prepared system
# ============================================================


@dataclass(slots=True)
class MAVESystem:
    """
    Fully assembled MAVE system.

    This object is the composition root exposed to train/evaluation scripts.
    It contains already connected components rather than an untyped dictionary.
    """

    config: Mapping[str, Any]

    escalation_backbone: LLMBackbone
    reaction_backbone: LLMBackbone
    single_step_model: SingleStepModel
    purchasable_db: PurchasableDatabase
    reward_fn: RewardFunction
    environment: RetroEnvironment

    query_factory: FeedbackQueryFactory
    cost_model: AcquisitionCostModel
    oracle: MultiFidelityOracleSystem

    escalation_policy: EscalationPolicy
    reaction_policy: ReactionPolicy

    planner: MCTSPlanner
    rollout_runner: PlanningRolloutRunner

    escalation_collector: EscalationGroupCollector
    reaction_collector: ReactionGroupCollector

    reference_escalation_policy: EscalationPolicy | None = None
    reference_reaction_policy: ReactionPolicy | None = None

    escalation_batch_provider: EscalationBatchProvider | None = None
    reaction_batch_provider: ReactionBatchProvider | None = None

    route_quality_evaluator: Any | None = None
    difficulty_labels: Mapping[str, str] | None = None

    checkpointables: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.escalation_policy.backbone is not self.escalation_backbone:
            raise ValueError(
                "escalation_policy must use escalation_backbone."
            )

        if self.reaction_policy.backbone is not self.reaction_backbone:
            raise ValueError(
                "reaction_policy must use reaction_backbone."
            )

        ensure_disjoint_parameters(
            self.escalation_trainable_parameters,
            self.reaction_trainable_parameters,
        )

    @property
    def escalation_trainable_parameters(self):
        return tuple(
            parameter
            for parameter in self.escalation_backbone.parameters()
            if parameter.requires_grad
        )

    @property
    def reaction_trainable_parameters(self):
        return tuple(
            parameter
            for parameter in self.reaction_backbone.parameters()
            if parameter.requires_grad
        )

    def require_training_ready(self) -> None:
        """
        Validate components required by scripts/train.py.
        """

        missing: list[str] = []

        if self.reference_escalation_policy is None:
            missing.append("reference_escalation_policy")

        if self.reference_reaction_policy is None:
            missing.append("reference_reaction_policy")

        if self.escalation_batch_provider is None:
            missing.append("escalation_batch_provider")

        if self.reaction_batch_provider is None:
            missing.append("reaction_batch_provider")

        if not self.escalation_trainable_parameters:
            missing.append("escalation_trainable_parameters")

        if not self.reaction_trainable_parameters:
            missing.append("reaction_trainable_parameters")

        if missing:
            raise SystemPreparationError(
                "Training system is incomplete. Missing: "
                + ", ".join(missing)
                + ". Supply them through SystemProviders. "
                "Provide the official experiment-specific training data and "
                "reference-policy construction through SystemProviders."
            )


# ============================================================
# Public preparation entry point
# ============================================================


def prepare_system(
    config: Mapping[str, Any],
    *,
    providers: SystemProviders | None = None,
) -> MAVESystem:
    """
    Construct and connect the complete MAVE planning system.

    Parameters
    ----------
    config:
        Configuration bundle created by scripts/train.py or
        scripts/evaluate.py.

    providers:
        Optional in-process experiment assets. Omitted fields remain explicit
        placeholders and fail only when their component is constructed.

    Assembly order
    --------------
    1. external model providers
    2. purchasable database and reward function
    3. single-step model and retrosynthesis environment
    4. independent trainable LLM backbones for mu and pi
    5. natural-language query factory
    6. L1-L4 oracle hierarchy, cost model, and cache
    7. MCTS inference planner
    8. rollout/collection interfaces used during training
    """

    providers = providers or SystemProviders()

    base_cfg = _section(config, "base")
    model_cfg = _section(config, "model")
    planner_cfg = _section(config, "planner")
    oracle_cfg = _section(config, "oracle")

    # --------------------------------------------------------
    # 1. Building-block database
    # --------------------------------------------------------

    purchasable_db = (
        providers.purchasable_db
        if providers.purchasable_db is not None
        else _prepare_purchasable_database(base_cfg)
    )

    # --------------------------------------------------------
    # 2. Single-step retrosynthesis model
    # --------------------------------------------------------

    single_step_model = _prepare_single_step_model(
        base_cfg=base_cfg,
        providers=providers,
    )

    # --------------------------------------------------------
    # 3. Reward and environment
    # --------------------------------------------------------

    reward_fn = (
        providers.reward_fn
        if providers.reward_fn is not None
        else _prepare_reward_function(base_cfg)
    )

    top_k = int(
        _get(
            planner_cfg,
            "search",
            "num_reaction_candidates",
            default=_get(
                base_cfg,
                "single_step",
                "top_k",
                default=50,
            ),
        )
    )

    environment = RetroEnvironment(
        single_step_model=single_step_model,
        purchasable_db=purchasable_db,
        reward_fn=reward_fn,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # 4. Parameter-isolated language-model policies
    # --------------------------------------------------------

    (
        escalation_backbone,
        reaction_backbone,
    ) = _prepare_policy_backbones(
        base_cfg=base_cfg,
        model_cfg=model_cfg,
        providers=providers,
    )

    policy_temperature = _policy_temperature(
        model_cfg
    )

    normalize_completion_logprob = bool(
        _get(
            model_cfg,
            "generation",
            "normalize_completion_logprob",
            default=True,
        )
    )

    escalation_policy = EscalationPolicy(
        escalation_backbone,
        temperature=policy_temperature,
        normalize_completion_logprob=normalize_completion_logprob,
    )

    reaction_policy = ReactionPolicy(
        reaction_backbone,
        temperature=policy_temperature,
        normalize_completion_logprob=normalize_completion_logprob,
    )

    (
        reference_escalation_policy,
        reference_reaction_policy,
    ) = _prepare_reference_policies(
        providers=providers,
        escalation_backbone=escalation_backbone,
        reaction_backbone=reaction_backbone,
        temperature=policy_temperature,
        normalize_completion_logprob=normalize_completion_logprob,
    )

    # --------------------------------------------------------
    # 5. Natural-language query interface
    # --------------------------------------------------------

    query_factory = _prepare_query_factory(
        oracle_cfg=oracle_cfg,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # 6. Multi-fidelity oracle hierarchy
    # --------------------------------------------------------

    if providers.round_trip_backend is None:
        raise SystemPreparationError(
            "No concrete round-trip backend was supplied. "
            "Provide SystemProviders.round_trip_backend."
        )

    if providers.yield_backend is None:
        raise SystemPreparationError(
            "No concrete yield backend was supplied. "
            "Provide SystemProviders.yield_backend."
        )

    depth_backend = (
        providers.depth_traversal_backend
        if providers.depth_traversal_backend is not None
        else DepthLimitedTraversalBackend(
            single_step_model=single_step_model,
            purchasable_db=purchasable_db,
            config=_prepare_traversal_config(
                oracle_cfg
            ),
        )
    )

    structural_oracle = StructuralOracle(
        backend=providers.round_trip_backend,
    )

    evaluative_oracle = EvaluativeOracle(
        yield_backend=providers.yield_backend,
        round_trip_backend=providers.round_trip_backend,
    )

    comparative_oracle = ComparativeOracle(
        yield_backend=providers.yield_backend,
        round_trip_backend=providers.round_trip_backend,
    )

    strategic_oracle = StrategicOracle(
        backend=depth_backend,
    )

    cost_model = AcquisitionCostModel()

    cache_enabled = bool(
        _get(
            oracle_cfg,
            "oracle",
            "feedback",
            "cache",
            "enabled",
            default=_get(
                base_cfg,
                "feedback",
                "cache_enabled",
                default=True,
            ),
        )
    )

    cache = (
        InMemoryFeedbackCache()
        if cache_enabled
        else NullFeedbackCache()
    )

    oracle = MultiFidelityOracleSystem(
        oracles=(
            structural_oracle,
            evaluative_oracle,
            comparative_oracle,
            strategic_oracle,
        ),
        cost_model=cost_model,
        query_factory=query_factory,
        cache=cache,
    )

    # --------------------------------------------------------
    # 7. MCTS inference planner
    # --------------------------------------------------------

    mcts_config = _prepare_mcts_config(
        planner_cfg=planner_cfg,
    )

    planner = MCTSPlanner(
        environment=environment,
        escalation_policy=escalation_policy,
        reaction_policy=reaction_policy,
        oracle=oracle,
        config=mcts_config,
    )

    # --------------------------------------------------------
    # 8. Training rollout interfaces
    # --------------------------------------------------------

    rollout_config = _prepare_rollout_config(
        config=config,
        planner_cfg=planner_cfg,
    )

    rollout_runner = PlanningRolloutRunner(
        environment=environment,
        escalation_policy=escalation_policy,
        reaction_policy=reaction_policy,
        oracle=oracle,
        config=rollout_config,
    )

    group_size = int(
        _get(
            _section(
                config,
                "mave_train",
                required=False,
            ),
            "rollout",
            "group_size",
            default=4,
        )
    )

    escalation_collector = EscalationGroupCollector(
        escalation_policy=escalation_policy,
        reaction_policy=reaction_policy,
        oracle=oracle,
        continuation_runner=rollout_runner,
        group_size=group_size,
        max_feedback_level=mcts_config.max_feedback_level,
    )

    reaction_collector = ReactionGroupCollector(
        reaction_policy=reaction_policy,
        continuation_runner=rollout_runner,
        group_size=group_size,
    )

    checkpointables = dict(
        providers.checkpointables
    )

    checkpointables.setdefault(
        "escalation_backbone",
        escalation_backbone,
    )

    checkpointables.setdefault(
        "reaction_backbone",
        reaction_backbone,
    )

    return MAVESystem(
        config=config,
        escalation_backbone=escalation_backbone,
        reaction_backbone=reaction_backbone,
        single_step_model=single_step_model,
        purchasable_db=purchasable_db,
        reward_fn=reward_fn,
        environment=environment,
        query_factory=query_factory,
        cost_model=cost_model,
        oracle=oracle,
        escalation_policy=escalation_policy,
        reaction_policy=reaction_policy,
        planner=planner,
        rollout_runner=rollout_runner,
        escalation_collector=escalation_collector,
        reaction_collector=reaction_collector,
        reference_escalation_policy=reference_escalation_policy,
        reference_reaction_policy=reference_reaction_policy,
        escalation_batch_provider=providers.escalation_batch_provider,
        reaction_batch_provider=providers.reaction_batch_provider,
        route_quality_evaluator=providers.route_quality_evaluator,
        difficulty_labels=providers.difficulty_labels,
        checkpointables=checkpointables,
    )


# ============================================================
# Component preparation
# ============================================================


def _prepare_single_step_model(
    *,
    base_cfg: Mapping[str, Any],
    providers: SystemProviders,
) -> SingleStepModel:
    if providers.single_step_model is not None:
        return providers.single_step_model

    if providers.single_step_backend is None:
        raise SystemPreparationError(
            "No concrete single-step retrosynthesis implementation was "
            "supplied. Provide either SystemProviders.single_step_model or "
            "SystemProviders.single_step_backend."
        )

    single_step_cfg = _get(
        base_cfg,
        "single_step",
        default={},
    )

    model_type = str(
        single_step_cfg.get(
            "type",
            "template_based",
        )
    )

    if model_type != "template_based":
        raise SystemPreparationError(
            "The current repository interface only assembles the "
            f"'template_based' single-step adapter, got {model_type!r}."
        )

    return TemplateBasedSingleStepModel(
        providers.single_step_backend,
        remove_duplicate_reactions=bool(
            single_step_cfg.get(
                "remove_duplicates",
                True,
            )
        ),
    )


def _prepare_purchasable_database(
    base_cfg: Mapping[str, Any],
) -> PurchasableDatabase:
    path_value = _get(
        base_cfg,
        "environment",
        "purchasable_database",
        "path",
        default=_get(
            base_cfg,
            "paths",
            "purchasable_db",
            default=None,
        ),
    )

    if not path_value:
        raise SystemPreparationError(
            "No purchasable building-block file/directory is configured."
        )

    path = Path(
        str(path_value)
    )

    if path.is_dir():
        candidates = [
            path / "smiles.txt",
            path / "emolecules.txt",
            path / "purchasable.txt",
            path / "smiles.csv",
            path / "emolecules.csv",
        ]

        existing = [
            item
            for item in candidates
            if item.is_file()
        ]

        if len(existing) != 1:
            raise SystemPreparationError(
                f"Purchasable database path {path} is a directory. "
                "Expected exactly one recognized file among: "
                + ", ".join(
                    item.name
                    for item in candidates
                )
                + "."
            )

        path = existing[0]

    if not path.is_file():
        raise SystemPreparationError(
            f"Purchasable database file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix in {
        ".txt",
        ".smi",
        ".smiles",
    }:
        return InMemoryPurchasableDatabase.from_txt(
            path
        )

    if suffix == ".csv":
        return InMemoryPurchasableDatabase.from_csv(
            path,
            smiles_column="smiles",
        )

    if suffix == ".tsv":
        return InMemoryPurchasableDatabase.from_csv(
            path,
            smiles_column="smiles",
            delimiter="\t",
        )

    raise SystemPreparationError(
        "Unsupported purchasable database format "
        f"{suffix!r} for {path}. Use txt/smi/smiles/csv/tsv."
    )


def _prepare_reward_function(
    base_cfg: Mapping[str, Any],
) -> RewardFunction:
    reward_cfg = _get(
        base_cfg,
        "environment",
        "reward",
        default={},
    )

    reward_type = reward_cfg.get(
        "type"
    )

    if reward_type is None:
        raise SystemPreparationError(
            "environment.reward.type is null. The paper does not uniquely "
            "specify the numerical R(s,a), so prepare_system() will not "
            "silently choose one. Set a concrete reward configuration or "
            "supply SystemProviders.reward_fn."
        )

    normalized = str(
        reward_type
    ).strip().lower()

    if normalized in {
        "sparse_terminal",
        "sparse_terminal_reward",
    }:
        success = reward_cfg.get(
            "terminal_success_reward"
        )
        failure = reward_cfg.get(
            "terminal_failure_reward"
        )
        intermediate = reward_cfg.get(
            "intermediate_reward"
        )

        if (
            success is None
            or failure is None
            or intermediate is None
        ):
            raise SystemPreparationError(
                "Sparse terminal reward requires "
                "terminal_success_reward, terminal_failure_reward, and "
                "intermediate_reward."
            )

        return SparseTerminalReward(
            success_reward=float(
                success
            ),
            step_reward_value=float(
                intermediate
            ),
            failure_reward_value=float(
                failure
            ),
        )

    if normalized == "zero":
        return ZeroReward()

    raise SystemPreparationError(
        f"Unknown reward type {reward_type!r}."
    )


def _prepare_backbone(
    *,
    base_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
) -> LLMBackbone:
    model = _get(
        model_cfg,
        "model",
        default={},
    )

    pretrained_name = model.get(
        "pretrained_name"
    )

    if not pretrained_name:
        raise SystemPreparationError(
            "model.model.pretrained_name is required."
        )

    dtype = model.get(
        "dtype",
        _get(
            base_cfg,
            "runtime",
            "dtype",
            default="bfloat16",
        ),
    )

    device = _get(
        base_cfg,
        "runtime",
        "device",
        default="cuda",
    )

    # LLMBackbone expects a transformers device_map. For CUDA the safest
    # repository default is "auto"; explicit provider code may override the
    # entire backbone when a different distributed-loading scheme is needed.
    device_map: str | None

    if str(device).lower() == "cpu":
        device_map = None
    else:
        device_map = "auto"

    tokenizer_cfg = model.get(
        "tokenizer",
        {}
    )

    tokenizer_kwargs: dict[str, Any] = {}

    truncation_side = tokenizer_cfg.get(
        "truncation_side"
    )

    if truncation_side is not None:
        tokenizer_kwargs[
            "truncation_side"
        ] = truncation_side

    return LLMBackbone(
        pretrained_name=str(
            pretrained_name
        ),
        dtype=str(dtype),
        device_map=device_map,
        tokenizer_kwargs=tokenizer_kwargs,
    )


def _prepare_policy_backbones(
    *,
    base_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    providers: SystemProviders,
) -> tuple[LLMBackbone, LLMBackbone]:
    shared_backbone = bool(
        _get(
            model_cfg,
            "policies",
            "shared_backbone",
            default=False,
        )
    )

    if shared_backbone:
        raise SystemPreparationError(
            "policies.shared_backbone=true is incompatible with the paper's "
            "alternating optimization: mu and pi require disjoint trainable "
            "parameters. Set it to false."
        )

    if providers.backbone is not None:
        raise SystemPreparationError(
            "SystemProviders.backbone is a deprecated shared trainable "
            "backbone. Supply escalation_backbone and reaction_backbone "
            "separately."
        )

    escalation_backbone = (
        providers.escalation_backbone
        if providers.escalation_backbone is not None
        else _prepare_backbone(
            base_cfg=base_cfg,
            model_cfg=model_cfg,
        )
    )
    reaction_backbone = (
        providers.reaction_backbone
        if providers.reaction_backbone is not None
        else _prepare_backbone(
            base_cfg=base_cfg,
            model_cfg=model_cfg,
        )
    )

    try:
        ensure_disjoint_parameters(
            (
                parameter
                for parameter in escalation_backbone.parameters()
                if parameter.requires_grad
            ),
            (
                parameter
                for parameter in reaction_backbone.parameters()
                if parameter.requires_grad
            ),
        )
    except ValueError as error:
        raise SystemPreparationError(str(error)) from error

    return escalation_backbone, reaction_backbone


def _prepare_reference_policies(
    *,
    providers: SystemProviders,
    escalation_backbone: LLMBackbone,
    reaction_backbone: LLMBackbone,
    temperature: float,
    normalize_completion_logprob: bool,
) -> tuple[
    EscalationPolicy | None,
    ReactionPolicy | None,
]:
    reference_backbone = (
        providers.reference_backbone
    )

    if reference_backbone is None:
        return None, None

    reference_parameter_ids = {
        id(parameter)
        for parameter in reference_backbone.parameters()
    }
    trainable_parameter_ids = {
        id(parameter)
        for backbone in (
            escalation_backbone,
            reaction_backbone,
        )
        for parameter in backbone.parameters()
    }

    if reference_parameter_ids.intersection(trainable_parameter_ids):
        raise SystemPreparationError(
            "reference_backbone must not share parameter objects with either "
            "trainable policy backbone."
        )

    reference_backbone.freeze()
    reference_backbone.eval()

    return (
        EscalationPolicy(
            reference_backbone,
            temperature=temperature,
            normalize_completion_logprob=normalize_completion_logprob,
        ),
        ReactionPolicy(
            reference_backbone,
            temperature=temperature,
            normalize_completion_logprob=normalize_completion_logprob,
        ),
    )


def _prepare_query_factory(
    *,
    oracle_cfg: Mapping[str, Any],
    top_k: int,
) -> FeedbackQueryFactory:
    setting = str(
        _get(
            oracle_cfg,
            "oracle",
            "setting",
            default="hierarchy",
        )
    )

    if setting != "hierarchy":
        raise SystemPreparationError(
            "The new natural-language query protocol currently preserves "
            "the global L1-L4 hierarchy. The 'yield_only' ablation remaps "
            "reaction_yield to a one-level setting and therefore needs its "
            "own explicit query-level adapter; prepare_system() refuses to "
            "silently change the hierarchy."
        )

    return FeedbackQueryFactory(
        FeedbackQueryFactoryConfig(
            max_candidates=top_k,
            include_cost_hints=True,
        )
    )


def _prepare_traversal_config(
    oracle_cfg: Mapping[str, Any],
) -> TraversalConfig:
    strategic_cfg = _get(
        oracle_cfg,
        "oracle",
        "hierarchy",
        "L4",
        "traversal",
        default={},
    )

    return TraversalConfig(
        max_depth=int(
            strategic_cfg.get(
                "max_depth",
                3,
            )
        ),
        max_expansions=int(
            strategic_cfg.get(
                "max_expansions",
                50,
            )
        ),
        top_k_per_node=int(
            strategic_cfg.get(
                "top_k_per_node",
                50,
            )
        ),
        goal_top_n=int(
            strategic_cfg.get(
                "goal_top_n",
                3,
            )
        ),
        strategy_top_n=int(
            strategic_cfg.get(
                "strategy_top_n",
                3,
            )
        ),
        max_strategy_length=int(
            strategic_cfg.get(
                "max_strategy_length",
                3,
            )
        ),
        min_strategy_support=int(
            strategic_cfg.get(
                "min_strategy_support",
                2,
            )
        ),
    )


def _prepare_mcts_config(
    *,
    planner_cfg: Mapping[str, Any],
) -> MCTSConfig:
    cpuct = _get(
        planner_cfg,
        "selection",
        "cpuct",
        default=None,
    )

    if cpuct is None:
        raise SystemPreparationError(
            "planner.selection.cpuct is null. The paper does not report "
            "c_puct, so set it explicitly as a reproduction choice."
        )

    feedback_budget = _get(
        planner_cfg,
        "feedback_budget",
        "max_cost",
        default=None,
    )

    return MCTSConfig(
        max_single_step_calls=int(
            _get(
                planner_cfg,
                "search",
                "max_single_step_calls",
                default=100,
            )
        ),
        top_k=int(
            _get(
                planner_cfg,
                "search",
                "num_reaction_candidates",
                default=50,
            )
        ),
        max_feedback_level=int(
            _get(
                planner_cfg,
                "escalation",
                "max_level",
                default=4,
            )
        ),
        gamma=float(
            _get(
                planner_cfg,
                "backup",
                "discount_factor",
                default=0.95,
            )
        ),
        cpuct=float(
            cpuct
        ),
        feedback_budget=(
            None
            if feedback_budget is None
            else float(
                feedback_budget
            )
        ),
        deterministic_policies=True,
    )


def _prepare_rollout_config(
    *,
    config: Mapping[str, Any],
    planner_cfg: Mapping[str, Any],
) -> PlanningRolloutConfig:
    mave_cfg = _section(
        config,
        "mave_train",
        required=False,
    )

    gamma = float(
        _get(
            mave_cfg,
            "return",
            "gamma",
            default=_get(
                planner_cfg,
                "backup",
                "discount_factor",
                default=0.95,
            ),
        )
    )

    return PlanningRolloutConfig(
        max_steps=int(
            _get(
                mave_cfg,
                "rollout",
                "max_steps",
                default=50,
            )
        ),
        max_single_step_calls=int(
            _get(
                planner_cfg,
                "search",
                "max_single_step_calls",
                default=100,
            )
        ),
        top_k=int(
            _get(
                planner_cfg,
                "search",
                "num_reaction_candidates",
                default=50,
            )
        ),
        max_feedback_level=int(
            _get(
                planner_cfg,
                "escalation",
                "max_level",
                default=4,
            )
        ),
        gamma=gamma,
        feedback_budget=_optional_float(
            _get(
                planner_cfg,
                "feedback_budget",
                "max_cost",
                default=None,
            )
        ),
    )


def _policy_temperature(
    model_cfg: Mapping[str, Any],
) -> float:
    value = _get(
        model_cfg,
        "generation",
        "temperature",
        default=None,
    )

    if value is None:
        # This is a scoring temperature, not free-form generation.
        # Keeping 1.0 preserves the unscaled model log-probabilities.
        return 1.0

    value = float(
        value
    )

    if value <= 0:
        raise SystemPreparationError(
            "generation.temperature must be positive."
        )

    return value


# ============================================================
# Configuration helpers
# ============================================================


def _section(
    config: Mapping[str, Any],
    name: str,
    *,
    required: bool = True,
) -> Mapping[str, Any]:
    value = config.get(
        name
    )

    if value is None:
        if required:
            raise SystemPreparationError(
                f"Missing configuration section {name!r}."
            )
        return {}

    if not isinstance(
        value,
        Mapping,
    ):
        raise TypeError(
            f"Configuration section {name!r} must be a mapping."
        )

    return value


def _get(
    mapping: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = mapping

    for key in keys:
        if not isinstance(
            current,
            Mapping,
        ):
            return default

        if key not in current:
            return default

        current = current[
            key
        ]

    return current


def _optional_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    return float(
        value
    )
