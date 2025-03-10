import os
import pathlib
import shutil
from itertools import chain

import nox

PROJECT = "bmi_example_agentpy"
PACKAGE = "diffusion"
HERE = pathlib.Path(__file__)
ROOT = HERE.parent
PATHS = [PACKAGE, "examples", "tests", HERE.name]


@nox.session()
def test(session: nox.Session) -> None:
    """Run the tests."""
    session.install(".[testing]")

    args = [
        "--cov",
        PACKAGE,
        "-vvv",
    ] + session.posargs

    if "CI" in os.environ:
        args.append(f"--cov-report=xml:{ROOT.absolute()!s}/coverage.xml")
    session.run("pytest", *args)

    if "CI" not in os.environ:
        session.run("coverage", "report", "--ignore-errors", "--show-missing")


@nox.session(name="test-bmi")
def test_bmi(session: nox.Session) -> None:
    """Test the Basic Model Interface."""
    session.install(".[testing]")
    session.run(
        "bmi-test",
        f"{PACKAGE}:BmiDiffusion",
        "--config-file",
        f"{ROOT}/examples/config.yaml",
        "--root-dir",
        "examples",
        "-vvv",
    )


@nox.session(name="run-examples")
def run_examples(session: nox.Session):
    """Run Python script examples."""
    session.install(".[examples]")
    session.cd(f"{ROOT}/examples")
    session.run("python", "run-bmi-model.py")


@nox.session(name="check-notebooks")
def check_notebooks(session: nox.Session) -> None:
    """Run the example notebooks."""
    session.install(".[testing,examples]")

    args = [
        "--nbmake",
        "--nbmake-kernel=python3",
        "--nbmake-timeout=3000",
        "-vvv",
    ] + session.posargs

    session.cd(f"{ROOT}/examples")
    session.run("pytest", *args)


@nox.session
def format(session: nox.Session) -> None:
    """Clean lint and assert style."""
    session.install(".[format]")

    if session.posargs:
        black_args = session.posargs
    else:
        black_args = []

    session.run("black", *black_args, *PATHS)
    session.run("isort", *PATHS)
    session.run("flake8", *PATHS)


@nox.session
def release(session):
    """Tag and build a new version."""
    session.install("zest.releaser")
    session.run("fullrelease")


@nox.session(python=False)
def clean(session):
    """Remove virtual environments, build files, and caches."""
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree(f"{PROJECT}.egg-info", ignore_errors=True)
    shutil.rmtree(".pytest_cache", ignore_errors=True)
    if os.path.exists(".coverage"):
        os.remove(".coverage")
    for p in chain(ROOT.rglob("*.py[co]"), ROOT.rglob("__pycache__")):
        if p.is_dir():
            p.rmdir()
        else:
            p.unlink()


@nox.session(python=False)
def nuke(session):
    """Clean and also remove the .nox directory."""
    clean(session)
    shutil.rmtree(".nox", ignore_errors=True)
