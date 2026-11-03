import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestQualityGates(unittest.TestCase):

    def test_ci_develop_branch_setup_is_idempotent(self):
        workflow = (REPO_ROOT / ".github/workflows/alltests.yml").read_text()
        branch_command = next(
            line.strip()
            for line in workflow.splitlines()
            if "git branch develop refs/remotes/origin/develop" in line
        )
        commands = [shlex.split(command) for command in branch_command.split(" || ")]

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "develop"], check=True)
            subprocess.run(
                [
                    "git", "-C", str(repo), "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid", "commit",
                    "-q", "--allow-empty", "-m", "initial",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "update-ref", "refs/remotes/origin/develop", "HEAD"],
                check=True,
            )

            self._run_shell_or(commands, repo)
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", "--detach"], check=True)
            subprocess.run(["git", "-C", str(repo), "branch", "-D", "develop"], check=True)
            self._run_shell_or(commands, repo)

            result = subprocess.run(
                ["git", "-C", str(repo), "show-ref", "--verify", "refs/heads/develop"]
            )
            self.assertEqual(result.returncode, 0)

    @staticmethod
    def _run_shell_or(commands, cwd):
        for command in commands:
            result = subprocess.run(command, cwd=cwd)
            if result.returncode == 0:
                return
        raise AssertionError("workflow branch setup command failed")

    def test_prepush_uses_read_only_strict_checks(self):
        makefile = (REPO_ROOT / "makefile").read_text()
        recipe = makefile.split("\nprepush:\n", 1)[1].split("\n\n", 1)[0]

        self.assertIn("$(MAKE) check_format", recipe)
        self.assertNotIn("$(MAKE) format", recipe)
        self.assertIn("$(MAKE) check_notebook_execution_changed STRICT=--strict", recipe)
        self.assertIn("$(MAKE) check_ref_style_changed STRICT=--strict", recipe)

    @unittest.skipIf(os.name == "nt" or shutil.which("sh") is None, "POSIX hook")
    def test_prepush_hook_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_make = Path(directory) / "make"
            fake_make.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
            fake_make.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{directory}:{env['PATH']}"

            result = subprocess.run(
                ["sh", str(REPO_ROOT / ".githooks/pre-push")],
                cwd=REPO_ROOT,
                env=env,
            )

        self.assertEqual(result.returncode, 17)

    @unittest.skipIf(
        os.name == "nt" or not shutil.which("bash") or not shutil.which("make"),
        "POSIX quality gates",
    )
    def test_ci_hook_gates(self):
        workflow = (REPO_ROOT / ".github/workflows/alltests.yml").read_text()
        step = workflow.split("- name: Check changed notebooks and references\n", 1)[1]
        script = textwrap.dedent(step.split("        run: |\n", 1)[1].split("\n  #", 1)[0])
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "scripts").mkdir()
            (repo / "demos").mkdir()
            (repo / "demos" / "legacy.ipynb").write_text(json.dumps({"cells": [{
                "cell_type": "code", "source": ["1 + 1"], "outputs": [],
                "execution_count": 5,
            }]}), encoding="utf-8")
            for name in ("check_notebook_execution.py", "check_ref_style.py"):
                shutil.copyfile(REPO_ROOT / "scripts" / name, repo / "scripts" / name)
            # Use the real gate and validators; stub unrelated checks.
            (repo / "makefile").write_text(
                (REPO_ROOT / "makefile").read_text() + "\ncheck_format check tests_fast:\n\t@true\n",
                encoding="utf-8",
            )

            def git(*args):
                return subprocess.run(
                    ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
                ).stdout.strip()

            git("init", "-q", "-b", "develop")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.invalid")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/develop", base)
            env = dict(os.environ, PYTHON=sys.executable, QUALITY_BASE=base, PYTHONDONTWRITEBYTECODE="1")
            notebook = repo / "demos" / "example.ipynb"
            reference = repo / "example.md"
            for category, count, citation in (
                (None, 1, False),
                ("execution-count-not-sequential", 2, False),
                ("undefined-citation", 1, True),
            ):
                with self.subTest(category=category):
                    notebook.write_text(json.dumps({"cells": [{
                        "cell_type": "code", "source": ["1 + 1"], "outputs": [],
                        "execution_count": count,
                    }]}), encoding="utf-8")
                    reference.write_text("See [1].\n" if citation else "Example.\n", encoding="utf-8")
                    git("add", ".")
                    git("commit", "-qm", "example")
                    commands = [
                        (["bash", "-c", script], base),  # push to develop
                        (["bash", "-c", script], ""),
                        (["bash", "-c", script], "0" * 40),
                        (["sh", str(REPO_ROOT / ".githooks/pre-push")], base),
                    ]
                    for index, (command, comparison) in enumerate(commands):
                        if index == 1:
                            git("branch", "-f", "feature", "HEAD")
                            git("checkout", "-q", "feature")
                            git("branch", "-f", "develop", base)
                        result = subprocess.run(
                            command, cwd=repo, env=dict(env, QUALITY_BASE=comparison),
                            capture_output=True, text=True,
                        )
                        self.assertEqual(result.returncode == 0, category is None, result.stdout + result.stderr)
                        if category:
                            self.assertIn(category, result.stdout)
                    self.assertEqual(git("status", "--porcelain"), "")
                    git("branch", "-f", "develop", "HEAD")
                    git("checkout", "-q", "develop")


if __name__ == "__main__":
    unittest.main()
