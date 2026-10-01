from __future__ import annotations

import unittest
from unittest.mock import patch

from mave.chemistry.molecule import Molecule
from mave.chemistry.reaction import Reaction
from mave.core.types import FeedbackLevel, PlanningState, ReactionCandidate
from mave.environment.state import RetroState
from mave.oracle.backends.round_trip import CallableRoundTripBackend
from mave.oracle.context import (
    build_oracle_context,
    native_candidates,
    native_selected_molecule,
)
from mave.oracle.query_factory import FeedbackQueryFactory
from mave.oracle.structural import StructuralOracle


class OracleContextAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        # Keep this contract test independent of the optional RDKit runtime.
        canonicalize = patch(
            "mave.chemistry.molecule.canonicalize_smiles",
            side_effect=lambda smiles: smiles,
        )
        self.addCleanup(canonicalize.stop)
        canonicalize.start()

        self.product = Molecule("CCO")
        self.reaction = Reaction(
            product=self.product,
            reactants=(Molecule("C"), Molecule("CO")),
            score=0.8,
        )
        self.state = RetroState.initial(self.product)

    def test_builds_core_context_and_preserves_backend_objects(self) -> None:
        context = build_oracle_context(
            state=self.state,
            selected_index=0,
            candidates=(self.reaction,),
            current_level=FeedbackLevel.L1,
            context_id="test-context",
        )

        self.assertIsInstance(context.state, PlanningState)
        self.assertIsInstance(context.candidates[0], ReactionCandidate)
        self.assertEqual(context.selected_molecule, "CCO")
        self.assertEqual(context.current_level, FeedbackLevel.L1)
        self.assertEqual(context.effective_context_id, "test-context")

        self.assertIs(native_selected_molecule(context), self.product)
        self.assertEqual(native_candidates(context), (self.reaction,))

    def test_query_factory_uses_native_reaction_for_execution_payload(self) -> None:
        context = build_oracle_context(
            state=self.state,
            selected_index=0,
            candidates=(self.reaction,),
            current_level=FeedbackLevel.L1,
        )

        queries = FeedbackQueryFactory().build_next_level_queries(
            context=context,
            current_level=FeedbackLevel.L1,
        )

        self.assertTrue(queries)
        self.assertTrue(
            all(query.payload["reaction"] is self.reaction for query in queries)
        )

    def test_structural_backend_receives_native_chemistry_objects(self) -> None:
        context = build_oracle_context(
            state=self.state,
            selected_index=0,
            candidates=(self.reaction,),
        )
        query = FeedbackQueryFactory().build_next_level_queries(
            context=context,
            current_level=FeedbackLevel.L0,
        )[0]
        received: dict[str, object] = {}

        def activity_assessment(product: object, candidates: object) -> str:
            received["product"] = product
            received["candidates"] = candidates
            return "plausible"

        oracle = StructuralOracle(
            backend=CallableRoundTripBackend(
                functions={"activity_assessment": activity_assessment}
            )
        )
        oracle.query(query, context)

        self.assertIs(received["product"], self.product)
        self.assertEqual(received["candidates"], (self.reaction,))

    def test_rejects_candidates_for_a_different_product(self) -> None:
        other_reaction = Reaction(
            product=Molecule("CC"),
            reactants=(Molecule("C"),),
        )

        with self.assertRaisesRegex(ValueError, "selected molecule"):
            build_oracle_context(
                state=self.state,
                selected_index=0,
                candidates=(other_reaction,),
            )


if __name__ == "__main__":
    unittest.main()
