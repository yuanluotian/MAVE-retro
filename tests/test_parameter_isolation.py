from __future__ import annotations

import unittest

from mave.training.parameter_isolation import (
    activate_parameter_set,
    ensure_disjoint_parameters,
    optimizer_parameters,
)


class _Parameter:
    def __init__(self, requires_grad: bool = True) -> None:
        self.requires_grad = requires_grad

    def requires_grad_(self, enabled: bool) -> "_Parameter":
        self.requires_grad = enabled
        return self


class _Optimizer:
    def __init__(self, *groups: list[_Parameter]) -> None:
        self.param_groups = [
            {"params": group}
            for group in groups
        ]


class ParameterIsolationTest(unittest.TestCase):
    def test_rejects_parameters_shared_by_two_policies(self) -> None:
        shared = _Parameter()

        with self.assertRaisesRegex(ValueError, "disjoint"):
            ensure_disjoint_parameters(
                (shared,),
                (shared,),
            )

    def test_phase_activation_freezes_only_inactive_policy(self) -> None:
        escalation = (_Parameter(False), _Parameter(False))
        reaction = (_Parameter(True),)

        activate_parameter_set(
            active=escalation,
            inactive=reaction,
        )

        self.assertTrue(all(item.requires_grad for item in escalation))
        self.assertTrue(all(not item.requires_grad for item in reaction))

        activate_parameter_set(
            active=reaction,
            inactive=escalation,
        )

        self.assertTrue(all(not item.requires_grad for item in escalation))
        self.assertTrue(all(item.requires_grad for item in reaction))

    def test_optimizer_parameters_follow_exact_param_groups(self) -> None:
        first = _Parameter()
        second = _Parameter()
        optimizer = _Optimizer([first], [second])

        self.assertEqual(
            optimizer_parameters(optimizer),
            (first, second),
        )

    def test_rejects_duplicate_parameter_within_optimizer(self) -> None:
        parameter = _Parameter()
        optimizer = _Optimizer([parameter], [parameter])

        with self.assertRaisesRegex(ValueError, "duplicates"):
            optimizer_parameters(optimizer)


if __name__ == "__main__":
    unittest.main()
