from __future__ import annotations

import ast
import unittest
from pathlib import Path


class PolicyBackboneWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repository_root = Path(__file__).resolve().parents[1]
        cls.interface_path = (
            cls.repository_root / "src" / "mave" / "mave" / "interface.py"
        )
        cls.interface_tree = ast.parse(
            cls.interface_path.read_text(encoding="utf-8"),
            filename=str(cls.interface_path),
        )

    def test_prepare_system_wires_distinct_policy_backbones(self) -> None:
        prepare_system = next(
            node
            for node in self.interface_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prepare_system"
        )
        expected = {
            "EscalationPolicy": "escalation_backbone",
            "ReactionPolicy": "reaction_backbone",
        }

        for constructor, backbone_name in expected.items():
            calls = [
                node
                for node in ast.walk(prepare_system)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == constructor
            ]
            self.assertEqual(len(calls), 1, constructor)
            self.assertIsInstance(calls[0].args[0], ast.Name, constructor)
            self.assertEqual(calls[0].args[0].id, backbone_name, constructor)

    def test_official_config_disables_shared_trainable_backbone(self) -> None:
        config = (
            self.repository_root / "configs" / "model" / "llama3_8b.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("shared_backbone: false", config)
        self.assertNotIn("shared_backbone: true", config)


if __name__ == "__main__":
    unittest.main()
