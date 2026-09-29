from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence,
)

from mave_repro.chemistry.molecule import (
    Molecule,
)
from mave_repro.chemistry.reaction import (
    Reaction,
)
from mave_repro.core.registry import (
    MODEL_REGISTRY,
)


# ============================================================
# Raw backend representation
# ============================================================


@dataclass(frozen=True, slots=True)
class RawSingleStepPrediction:
    """
    Backend-independent single-step prediction.
    """

    reactants: tuple[str, ...]

    score: float

    template_id: str | None = None
    reaction_id: str | None = None

    metadata: Mapping[str, Any] | None = None


# ============================================================
# Backend protocol
# ============================================================


class TemplatePredictorBackend(
    Protocol
):
    """
    Minimal interface expected from an external template-based
    single-step retrosynthesis predictor.
    """

    def predict(
        self,
        product_smiles: str,
        *,
        top_k: int,
    ) -> Sequence[Any]:
        ...


# ============================================================
# Abstract model
# ============================================================


class SingleStepModel(ABC):
    """
    Interface for single-step retrosynthesis models.
    """

    @abstractmethod
    def predict(
        self,
        product_smiles: str,
        *,
        top_k: int = 50,
    ) -> tuple[Reaction, ...]:
        ...


# ============================================================
# Template-based adapter
# ============================================================


@MODEL_REGISTRY.register(
    "template_based"
)
class TemplateBasedSingleStepModel(
    SingleStepModel
):
    """
    Adapter around a template-based retrosynthesis predictor.

    The actual Chen et al. (2020)-style model/checkpoint is kept
    outside this class because the paper does not uniquely specify
    all implementation details needed to reconstruct it.

    Parameters
    ----------
    backend:
        Object exposing predict(product_smiles, top_k=...).

    parser:
        Optional function converting one backend output to
        RawSingleStepPrediction.
    """

    def __init__(
        self,
        backend: TemplatePredictorBackend,
        *,
        parser: Callable[
            [Any],
            RawSingleStepPrediction
        ] | None = None,
        remove_duplicate_reactions: bool = True,
    ) -> None:

        self.backend = backend

        self.parser = (
            parser
            or self._default_parser
        )

        self.remove_duplicate_reactions = (
            remove_duplicate_reactions
        )

    def predict(
        self,
        product_smiles: str,
        *,
        top_k: int = 50,
    ) -> tuple[Reaction, ...]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        product = Molecule(
            product_smiles
        )

        raw_predictions = (
            self.backend.predict(
                product.canonical_smiles,
                top_k=top_k,
            )
        )

        reactions: list[
            Reaction
        ] = []

        seen: set[
            tuple[
                str,
                tuple[str, ...],
            ]
        ] = set()

        for raw in raw_predictions:

            parsed = self.parser(
                raw
            )

            reaction = Reaction.from_smiles(
                product=(
                    product.canonical_smiles
                ),
                reactants=(
                    parsed.reactants
                ),
                score=float(
                    parsed.score
                ),
                template_id=(
                    parsed.template_id
                ),
                reaction_id=(
                    parsed.reaction_id
                ),
                metadata=(
                    parsed.metadata
                    or {}
                ),
            )

            if (
                self.remove_duplicate_reactions
                and reaction.key in seen
            ):
                continue

            seen.add(
                reaction.key
            )

            reactions.append(
                reaction
            )

            if len(reactions) >= top_k:
                break

        # Preserve backend ranking unless its scores provide
        # an unambiguous ranking.
        return tuple(reactions)

    # ========================================================
    # Default output parser
    # ========================================================

    @staticmethod
    def _default_parser(
        raw: Any,
    ) -> RawSingleStepPrediction:
        """
        Accept a few common predictor output formats.

        Supported examples:

        {
            "reactants": ["CCO", "O"],
            "score": 0.91,
            "template_id": "123"
        }

        ("CCO.O", 0.91)

        RawSingleStepPrediction(...)
        """

        if isinstance(
            raw,
            RawSingleStepPrediction,
        ):
            return raw

        if isinstance(raw, Mapping):

            reactants = raw.get(
                "reactants"
            )

            score = raw.get(
                "score"
            )

            if reactants is None:
                raise ValueError(
                    "Prediction mapping is missing "
                    "'reactants'."
                )

            if score is None:
                raise ValueError(
                    "Prediction mapping is missing "
                    "'score'."
                )

            return RawSingleStepPrediction(
                reactants=(
                    TemplateBasedSingleStepModel
                    ._parse_reactants(
                        reactants
                    )
                ),
                score=float(score),
                template_id=(
                    raw.get(
                        "template_id"
                    )
                ),
                reaction_id=(
                    raw.get(
                        "reaction_id"
                    )
                ),
                metadata=dict(
                    raw.get(
                        "metadata",
                        {},
                    )
                ),
            )

        if (
            isinstance(raw, tuple)
            and len(raw) >= 2
        ):
            return RawSingleStepPrediction(
                reactants=(
                    TemplateBasedSingleStepModel
                    ._parse_reactants(
                        raw[0]
                    )
                ),
                score=float(
                    raw[1]
                ),
                template_id=(
                    str(raw[2])
                    if len(raw) >= 3
                    and raw[2] is not None
                    else None
                ),
            )

        raise TypeError(
            "Unsupported single-step backend "
            f"prediction type: {type(raw)!r}"
        )

    @staticmethod
    def _parse_reactants(
        value: Any,
    ) -> tuple[str, ...]:

        if isinstance(
            value,
            str,
        ):
            reactants = tuple(
                item.strip()
                for item
                in value.split(".")
                if item.strip()
            )

        elif isinstance(
            value,
            Sequence,
        ):
            reactants = tuple(
                str(item).strip()
                for item in value
                if str(item).strip()
            )

        else:
            raise TypeError(
                "reactants must be a dot-separated "
                "SMILES string or sequence."
            )

        if not reactants:
            raise ValueError(
                "Prediction contains no reactants."
            )

        return reactants