"""
Nox creates a new virtual environment for each individual test.  Thus, it is important for to install all the packages needed for testing.  When using Nox, it will by default grab the current python version available in your environment and run testing with it.

Useful commands:

```console
nox --list                 # Lists out all the available sessions
nox -r -s pytest           # Run pytests
nox -r -s pytest_cov       # Run pytests with coverage report
nox -r -s coverage         # Run coverage (stand-alone)
nox -r -s scalene          # Profile the code

nox                        # Run all sessions
```

"""

from dataclasses import dataclass

import nox

from src.ci.utils import view_html


@dataclass
class config:
    pytest_cov_path: str = "save/pytest-cov"
    coverage_path: str = "save/coverage"


@nox.session
def pytest(session: nox.Session):
    """Run PyTests."""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("pytest", "-v")


@nox.session
def pytest_cov(session: nox.Session):
    """Run PyTests with coverage."""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("pytest", "--cov=./", f"--cov-report=html:{config.pytest_cov_path}")


@nox.session
def coverage(session: nox.Session):
    """Runs coverage pytests"""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "html", "-d", config.coverage_path)
    session.run("coverage", "report", "-m")


@nox.session
def scalene(session: nox.Session):
    """Profiles your selected code using scalene."""

    session.run("poetry", "install", "--with=dev", "--no-root")
    session.run("scalene", "-m", "pytest")
