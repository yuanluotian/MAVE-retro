from __future__ import annotations

import unittest
from pathlib import Path


class NoProviderFactoryTest(unittest.TestCase):
    def test_dynamic_provider_factory_is_not_exposed(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        paths = (
            repository_root / "README.md",
            repository_root / "src" / "mave" / "mave" / "interface.py",
            repository_root / "scripts" / "train.py",
            repository_root / "scripts" / "evaluate.py",
            repository_root / "scripts" / "eval_budget_scaling.py",
        )
        forbidden = (
            "providers:prepare_providers",
            "provider_factory",
            "provider-factory",
            "load_provider_factory",
        )

        for path in paths:
            content = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, content, f"{marker!r} remains in {path}")


if __name__ == "__main__":
    unittest.main()
