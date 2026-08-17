import ast
import unittest
from pathlib import Path


class PackageBoundaryTests(unittest.TestCase):
    def test_framework_does_not_depend_on_problem_model_or_experiment_layers(self) -> None:
        framework_directory = (
            Path(__file__).resolve().parents[2] / "src" / "groupopt" / "framework"
        )
        forbidden = (
            "groupopt.adapters",
            "groupopt.models",
            "groupopt.objectives",
            "groupopt.problems",
            "experiments",
        )
        violations: list[str] = []
        for path in sorted(framework_directory.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: tuple[str, ...] = ()
                if isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = (node.module,)
                for module in modules:
                    if module.startswith(forbidden):
                        violations.append(f"{path.name}:{node.lineno} imports {module}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
