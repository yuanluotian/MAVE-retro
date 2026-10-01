from __future__ import annotations

import ast
import unittest
from pathlib import Path


class OracleQueryCallOrderTest(unittest.TestCase):
    def test_planner_call_sites_use_query_then_context(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        expected_context_names = {
            "planning/mcts.py": "oracle_context",
            "planning/rollout.py": "context",
            "training/collector.py": "oracle_context",
        }

        for relative_path, context_name in expected_context_names.items():
            path = repository_root / "src" / "mave" / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "query"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "oracle"
            ]

            self.assertEqual(len(calls), 1, relative_path)
            call = calls[0]
            self.assertEqual(len(call.args), 2, relative_path)
            self.assertIsInstance(call.args[0], ast.Name, relative_path)
            self.assertEqual(call.args[0].id, "query", relative_path)
            self.assertIsInstance(call.args[1], ast.Name, relative_path)
            self.assertEqual(call.args[1].id, context_name, relative_path)


if __name__ == "__main__":
    unittest.main()
