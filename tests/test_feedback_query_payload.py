from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from mave.chemistry.molecule import Molecule
from mave.chemistry.reaction import Reaction
from mave.chemistry.route import SynthesisRoute
from mave.core.types import FeedbackLevel
from mave.environment.state import RetroState
from mave.oracle.backends.round_trip import CallableRoundTripBackend
from mave.oracle.backends.yield_model import CallableYieldModelBackend
from mave.oracle.base import FeedbackType
from mave.oracle.cache import make_cache_key
from mave.oracle.comparative import ComparativeOracle
from mave.oracle.context import build_oracle_context
from mave.oracle.evaluative import EvaluativeOracle
from mave.oracle.query_factory import FeedbackQueryFactory


class FeedbackQueryPayloadTest(unittest.TestCase):
    def setUp(self) -> None:
        canonicalize = patch(
            "mave.chemistry.molecule.canonicalize_smiles",
            side_effect=lambda smiles: smiles,
        )
        self.addCleanup(canonicalize.stop)
        canonicalize.start()

        self.product = Molecule("CCO")
        self.reaction_a = Reaction(
            product=self.product,
            reactants=(Molecule("C"), Molecule("CO")),
            score=0.8,
        )
        self.reaction_b = Reaction(
            product=self.product,
            reactants=(Molecule("CC"), Molecule("O")),
            score=0.6,
        )
        self.state = RetroState.initial(self.product)
        self.context = build_oracle_context(
            state=self.state,
            selected_index=0,
            candidates=(self.reaction_a, self.reaction_b),
        )

    def _query(self, current_level: FeedbackLevel, feedback_type: FeedbackType):
        queries = FeedbackQueryFactory().build_next_level_queries(
            context=self.context,
            current_level=current_level,
        )
        return next(
            query
            for query in queries
            if query.feedback_type == feedback_type.value
        )

    def test_cache_key_uses_semantic_key_and_payload(self) -> None:
        query = self._query(
            FeedbackLevel.L1,
            FeedbackType.REACTION_YIELD,
        )
        rewritten_question = replace(
            query,
            question="Equivalent wording must not change cache identity.",
        )
        changed_payload = replace(
            query,
            payload={**query.payload, "candidate_index": 999},
        )

        key = make_cache_key(query, self.context)
        self.assertEqual(
            key,
            make_cache_key(rewritten_question, self.context),
        )
        self.assertNotEqual(
            key,
            make_cache_key(changed_payload, self.context),
        )

    def test_evaluative_oracle_reads_reaction_from_payload(self) -> None:
        query = self._query(
            FeedbackLevel.L1,
            FeedbackType.REACTION_YIELD,
        )
        received: list[Reaction] = []
        oracle = EvaluativeOracle(
            yield_backend=CallableYieldModelBackend(
                lambda reaction: received.append(reaction) or 0.75
            ),
            round_trip_backend=CallableRoundTripBackend(functions={}),
        )

        result = oracle.query(query, self.context)

        self.assertEqual(received, [self.reaction_a])
        self.assertIn("0.750", result.content)

    def test_comparative_oracle_reads_reactions_from_payload(self) -> None:
        query = self._query(
            FeedbackLevel.L2,
            FeedbackType.REACTION_COMPARISON,
        )
        received: list[Reaction] = []

        def predict(reaction: Reaction) -> float:
            received.append(reaction)
            return 0.8 if reaction == self.reaction_a else 0.4

        oracle = ComparativeOracle(
            yield_backend=CallableYieldModelBackend(predict),
            round_trip_backend=CallableRoundTripBackend(functions={}),
        )

        result = oracle.query(query, self.context)

        self.assertEqual(received, [self.reaction_a, self.reaction_b])
        self.assertIn("Candidate reaction 1", result.content)

    def test_comparative_oracle_reads_routes_from_payload(self) -> None:
        route_a = self.state.route
        route_b = RetroState.initial(Molecule("CCN")).route
        context = build_oracle_context(
            state=self.state,
            selected_index=0,
            candidates=(self.reaction_a, self.reaction_b),
            metadata={"routes": (route_a, route_b)},
        )
        queries = FeedbackQueryFactory().build_next_level_queries(
            context=context,
            current_level=FeedbackLevel.L2,
        )
        query = next(
            item
            for item in queries
            if item.feedback_type == FeedbackType.ROUTE_COMPARISON.value
        )
        received: list[SynthesisRoute] = []

        def route_score(route: SynthesisRoute) -> float:
            received.append(route)
            return 0.9 if route is route_a else 0.3

        oracle = ComparativeOracle(
            yield_backend=CallableYieldModelBackend(lambda reaction: 0.5),
            round_trip_backend=CallableRoundTripBackend(
                functions={"route_score": route_score}
            ),
        )

        result = oracle.query(query, context)

        self.assertEqual(received, [route_a, route_b])
        self.assertIn("Partial route 1", result.content)


if __name__ == "__main__":
    unittest.main()
