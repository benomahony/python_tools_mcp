#!/usr/bin/env python3
"""Python Tools MCP Server - Opinionated Python development tools via MCP."""

import subprocess
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

__version__ = "0.2.0"

mcp = FastMCP("python-tools")


class CommandResult(BaseModel):
    """Result of a command execution."""

    returncode: int
    stdout: str
    stderr: str
    success: bool


def run_command(cmd: list[str], cwd: Path | None = None) -> CommandResult:
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            success=(result.returncode == 0),
        )
    except Exception as e:
        return CommandResult(
            returncode=1,
            stdout="",
            stderr=f"Failed to run command {' '.join(cmd)}: {e}",
            success=False,
        )


def get_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or .git."""
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current


def format_result(cmd: list[str], result: CommandResult) -> str:
    """Format command result for output."""
    output = f"Command: {' '.join(cmd)}\n"
    output += f"Exit code: {result.returncode}\n\n"

    if result.stdout:
        output += f"STDOUT:\n{result.stdout}\n"

    if result.stderr:
        output += f"STDERR:\n{result.stderr}\n"

    if not result.success:
        output += f"\n❌ Command failed with exit code {result.returncode}"
    else:
        output += "\n✅ Command completed successfully"

    return output


@mcp.tool()
def uv_add(
    packages: Annotated[list[str], Field(description="Package names to add")],
    dev: Annotated[
        bool, Field(default=False, description="Add as dev dependency")
    ] = False,
) -> str:
    """Add dependencies using uv."""
    project_root = get_project_root()
    dev_flag = ["--dev"] if dev else []
    cmd = ["uv", "add"] + dev_flag + packages
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_remove(
    packages: Annotated[list[str], Field(description="Package names to remove")],
    dev: Annotated[
        bool, Field(default=False, description="Remove from dev dependencies")
    ] = False,
) -> str:
    """Remove dependencies using uv."""
    project_root = get_project_root()
    dev_flag = ["--dev"] if dev else []
    cmd = ["uv", "remove"] + dev_flag + packages
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_sync(
    dev: Annotated[
        bool, Field(default=True, description="Include dev dependencies")
    ] = True,
    all_extras: Annotated[
        bool, Field(default=False, description="Install all optional dependencies")
    ] = False,
    all_groups: Annotated[
        bool, Field(default=False, description="Include all dependency groups")
    ] = False,
    all_packages: Annotated[
        bool, Field(default=False, description="Install all packages in pyproject.toml")
    ] = False,
    frozen: Annotated[
        bool, Field(default=False, description="Install exact versions from lockfile")
    ] = False,
) -> str:
    """Sync project dependencies with uv.

    Examples:
        Basic sync with dev dependencies:
            uv_sync()

        Frozen sync without dev dependencies:
            uv_sync(dev=False, frozen=True)

        Complete sync with all options:
            uv_sync(all_extras=True, all_groups=True, all_packages=True)
    """
    project_root = get_project_root()
    flags = []

    if dev:
        flags.append("--dev")
    else:
        flags.append("--no-dev")

    if all_extras:
        flags.append("--all-extras")

    if all_groups:
        flags.append("--all-groups")

    if all_packages:
        flags.append("--all-packages")

    if frozen:
        flags.append("--frozen")

    cmd = ["uv", "sync"] + flags
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uvx_run(
    tool: Annotated[str, Field(description="Tool to run")],
    args: Annotated[
        list[str],
        Field(default_factory=list, description="Arguments to pass to the tool"),
    ],
    with_deps: Annotated[
        list[str] | None,
        Field(default_factory=list, description="Additional dependencies to include"),
    ] = None,
) -> str:
    """Run a tool with uvx."""
    project_root = get_project_root()
    cmd = ["uvx"]

    # Add dependencies with --with flag if specified
    if with_deps:
        for dep in with_deps:
            cmd.extend(["--with", dep])

    cmd.append(tool)
    cmd.extend(args)

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def pyupgrade(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to process"
        ),
    ],
    target_version: Annotated[
        str,
        Field(
            default="py312", description="Target Python version (e.g., py311, py312)"
        ),
    ] = "py312",
) -> str:
    """Upgrade Python syntax using pyupgrade."""
    project_root = get_project_root()
    cmd = ["uvx", "pyupgrade", f"--{target_version}-plus"] + files
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def basedpyright(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to check"
        ),
    ],
) -> str:
    """Run basedpyright type checker."""
    project_root = get_project_root()

    cmd = ["uvx", "--with", "basedpyright", "basedpyright"] + files
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def ruff_check(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to check"
        ),
    ],
    fix: Annotated[
        bool, Field(default=False, description="Automatically fix issues")
    ] = False,
) -> str:
    """Run ruff linter."""
    project_root = get_project_root()
    fix_flag = ["--fix"] if fix else []
    cmd = ["uvx", "ruff", "check"] + fix_flag + files
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def ruff_format(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to format"
        ),
    ],
) -> str:
    """Format code with ruff."""
    project_root = get_project_root()
    cmd = ["uvx", "ruff", "format"] + files
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def pytest(
    files: Annotated[
        list[str],
        Field(default_factory=list, description="Test files or directories to run"),
    ],
    verbose: Annotated[
        bool, Field(default=False, description="Verbose output")
    ] = False,
    coverage: Annotated[
        bool, Field(default=False, description="Run with coverage")
    ] = False,
    parallel: Annotated[
        bool,
        Field(default=False, description="Run tests in parallel with pytest-xdist"),
    ] = False,
    markers: Annotated[
        str,
        Field(default="", description="Run tests matching markers (e.g., 'not slow')"),
    ] = "",
    with_deps: Annotated[
        list[str] | None,
        Field(default_factory=list, description="Additional pytest plugins to include"),
    ] = None,
) -> str:
    """Run tests with pytest.

    Examples:
        Basic usage:
            pytest(files=["."])

        With coverage:
            pytest(files=["."], coverage=True)

        With custom plugins:
            pytest(files=["."], with_deps=["pytest-asyncio", "pytest-benchmark"])
    """
    project_root = get_project_root()

    flags = []
    if verbose:
        flags.append("-v")
    if coverage:
        flags.extend(["--cov", "--cov-report=term-missing"])
    if parallel:
        flags.append("-n auto")
    if markers:
        flags.extend(["-m", markers])

    cmd = ["uvx"]

    # Add required plugins using --with flag
    if coverage:
        cmd.extend(["--with", "pytest-cov"])
    if parallel:
        cmd.extend(["--with", "pytest-xdist"])

    # Add custom plugins if specified
    if with_deps:
        for dep in with_deps:
            cmd.extend(["--with", dep])

    # Add the main command
    cmd.extend(["pytest"] + flags + files)

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_lock() -> str:
    """Update the lockfile with uv."""
    project_root = get_project_root()
    cmd = ["uv", "lock"]
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_list(
    packages: Annotated[
        list[str], Field(default_factory=list, description="Filter by these packages")
    ],
    outdated: Annotated[
        bool, Field(default=False, description="Show only outdated packages")
    ] = False,
    format_type: Annotated[
        str, Field(default="json", description="Output format (json, markdown, plain)")
    ] = "json",
) -> str:
    """List installed packages with uv."""
    project_root = get_project_root()
    cmd = ["uv", "list"]

    if outdated:
        cmd.append("--outdated")

    if format_type != "plain":
        cmd.extend(["--format", format_type])

    cmd.extend(packages)
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_tree(
    depth: Annotated[int, Field(default=3, description="Maximum depth to display")] = 3,
    package: Annotated[
        str, Field(default="", description="Show tree for specific package")
    ] = "",
) -> str:
    """Show dependency tree with uv."""
    project_root = get_project_root()
    cmd = ["uv", "tree", "--depth", str(depth)]
    if package:
        cmd.append(package)
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_export(
    format_type: Annotated[
        str,
        Field(
            default="requirements",
            description="Output format (requirements, pyproject, conda)",
        ),
    ] = "requirements",
    output: Annotated[
        str, Field(default="", description="Output file (default: stdout)")
    ] = "",
    include_dev: Annotated[
        bool, Field(default=True, description="Include dev dependencies")
    ] = True,
    include_packages: Annotated[
        list[str] | None,
        Field(default_factory=list, description="Only include specific packages"),
    ] = None,
    extras: Annotated[
        list[str] | None,
        Field(default_factory=list, description="Only include specific extras"),
    ] = None,
) -> str:
    """Export dependencies to various formats with uv."""
    project_root = get_project_root()
    cmd = ["uv", "export"]

    cmd.extend(["--format", format_type])

    if output:
        cmd.extend(["--output", output])

    if include_dev:
        cmd.append("--dev")

    if include_packages:
        for package in include_packages:
            cmd.extend(["--include-packages", package])

    if extras:
        for extra in extras:
            cmd.extend(["--extras", extra])

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_run(
    command: Annotated[str, Field(description="Command to run in project environment")],
    args: Annotated[
        list[str], Field(default_factory=list, description="Arguments for the command")
    ],
) -> str:
    """Run command in the project environment with uv."""
    project_root = get_project_root()
    cmd = ["uv", "run", command] + args
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_build(
    wheel: Annotated[bool, Field(default=True, description="Build wheel")] = True,
    sdist: Annotated[
        bool, Field(default=False, description="Build source distribution")
    ] = False,
) -> str:
    """Build the project with uv."""
    project_root = get_project_root()
    cmd = ["uv", "build"]

    if not wheel and not sdist:
        # Default to wheel if nothing specified
        wheel = True

    if wheel and not sdist:
        cmd.append("--wheel")
    elif sdist and not wheel:
        cmd.append("--sdist")
    # If both are True, build both (default behavior)

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def bandit(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to scan"
        ),
    ],
    skip_tests: Annotated[
        bool, Field(default=True, description="Skip test files")
    ] = True,
    confidence: Annotated[
        str, Field(default="medium", description="Confidence level (low, medium, high)")
    ] = "medium",
) -> str:
    """Run bandit security linter."""
    project_root = get_project_root()

    cmd = ["uvx", "--with", "bandit", "bandit", "-r"] + files
    if skip_tests:
        cmd.extend(["--skip", "B101"])

    # Use the correct confidence flag format
    cmd.extend(["-c", confidence])

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def vulture(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to check"
        ),
    ],
    min_confidence: Annotated[
        int, Field(default=80, description="Minimum confidence (0-100)")
    ] = 80,
) -> str:
    """Find dead code with vulture."""
    project_root = get_project_root()

    cmd = (
        ["uvx", "--with", "vulture", "vulture"]
        + files
        + ["--min-confidence", str(min_confidence)]
    )
    result = run_command(cmd, cwd=project_root)
    # Return success even if dead code is found - this is a report tool, not a failure
    if result.returncode != 0 and result.stderr.strip() == "":
        result.success = True
    return format_result(cmd, result)


@mcp.tool()
def mypy(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to check"
        ),
    ],
    strict: Annotated[
        bool, Field(default=False, description="Enable strict mode")
    ] = False,
    install_types: Annotated[
        bool, Field(default=False, description="Install missing type stubs")
    ] = False,
) -> str:
    """Run mypy type checker."""
    project_root = get_project_root()

    cmd = ["uvx", "mypy"]
    if strict:
        cmd.append("--strict")
    if install_types:
        cmd.append("--install-types")
    cmd.extend(files)

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def pre_commit(
    action: Annotated[
        str,
        Field(
            default="run", description="Action to perform (run, install, autoupdate)"
        ),
    ] = "run",
    all_files: Annotated[
        bool, Field(default=False, description="Run on all files")
    ] = False,
) -> str:
    """Run pre-commit hooks."""
    project_root = get_project_root()

    if action == "install":
        cmd = ["uvx", "pre-commit", "install"]
    elif action == "autoupdate":
        cmd = ["uvx", "pre-commit", "autoupdate"]
    else:  # run
        cmd = ["uvx", "pre-commit", "run"]
        if all_files:
            cmd.append("--all-files")

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def coverage_report(
    format_type: Annotated[
        str, Field(default="term", description="Report format (term, html, xml, json)")
    ] = "term",
    show_missing: Annotated[
        bool, Field(default=True, description="Show missing lines")
    ] = True,
) -> str:
    """Generate coverage report."""
    project_root = get_project_root()

    cmd = ["uvx", "coverage", "report"]
    if format_type == "html":
        cmd = ["uvx", "coverage", "html"]
    elif format_type == "xml":
        cmd = ["uvx", "coverage", "xml"]
    elif format_type == "json":
        cmd = ["uvx", "coverage", "json"]

    if format_type == "term" and show_missing:
        cmd.append("--show-missing")

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def pdoc(
    modules: Annotated[list[str], Field(description="Modules to document")],
    output_dir: Annotated[
        str, Field(default="docs", description="Output directory")
    ] = "docs",
    format_type: Annotated[
        str, Field(default="html", description="Output format (html, pdf)")
    ] = "html",
) -> str:
    """Generate API documentation with pdoc."""
    project_root = get_project_root()

    cmd = ["uvx", "pdoc"]
    if format_type == "html":
        cmd.extend(["--html", "--output-dir", output_dir])
    elif format_type == "pdf":
        cmd.extend(["--pdf"])

    cmd.extend(modules)
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def uv_python_list() -> str:
    """List available Python versions with uv."""
    cmd = ["uv", "python", "list"]
    result = run_command(cmd)
    return format_result(cmd, result)


@mcp.tool()
def uv_python_install(
    version: Annotated[
        str, Field(description="Python version to install (e.g., '3.12', '3.11.9')")
    ],
) -> str:
    """Install Python version with uv."""
    cmd = ["uv", "python", "install", version]
    result = run_command(cmd)
    return format_result(cmd, result)


@mcp.tool()
def pip_audit() -> str:
    """Audit dependencies for known security vulnerabilities."""
    project_root = get_project_root()
    cmd = ["uvx", "pip-audit"]
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def radon_complexity(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to analyze"
        ),
    ],
    min_complexity: Annotated[
        str,
        Field(default="B", description="Minimum complexity to show (A, B, C, D, E, F)"),
    ] = "B",
) -> str:
    """Analyze code complexity with radon."""
    project_root = get_project_root()

    cmd = (
        ["uvx", "--with", "radon", "radon", "cc"]
        + files
        + ["-s", f"--min={min_complexity}"]
    )
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def radon_maintainability(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to analyze"
        ),
    ],
    min_index: Annotated[
        str, Field(default="B", description="Minimum maintainability index to show")
    ] = "B",
) -> str:
    """Analyze maintainability index with radon."""
    project_root = get_project_root()

    cmd = (
        ["uvx", "--with", "radon", "radon", "mi"] + files + ["-s", f"--min={min_index}"]
    )
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def py_spy(
    command: Annotated[str, Field(description="Command to profile")],
    duration: Annotated[int, Field(default=10, description="Duration in seconds")] = 10,
    rate: Annotated[int, Field(default=100, description="Sampling rate (Hz)")] = 100,
    output: Annotated[
        str, Field(default="profile.svg", description="Output file")
    ] = "profile.svg",
) -> str:
    """Profile Python code with py-spy."""
    project_root = get_project_root()
    cmd = [
        "uvx",
        "py-spy",
        "record",
        "-o",
        output,
        "-d",
        str(duration),
        "-r",
        str(rate),
        "--",
        command,
    ]
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def safety_check() -> str:
    """Check dependencies for known security vulnerabilities with safety."""
    project_root = get_project_root()
    cmd = ["uvx", "safety", "check"]
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def interrogate(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to check"
        ),
    ],
    fail_under: Annotated[
        int, Field(default=80, description="Fail if coverage below this percentage")
    ] = 80,
    verbose: Annotated[
        bool, Field(default=False, description="Verbose output")
    ] = False,
) -> str:
    """Check docstring coverage with interrogate."""
    project_root = get_project_root()

    cmd = ["uvx", "interrogate", f"--fail-under={fail_under}"]
    if verbose:
        cmd.append("-vv")
    cmd.extend(files)

    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def semgrep(
    files: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["."], description="Files or directories to scan"
        ),
    ],
    config: Annotated[
        str,
        Field(
            default="auto",
            description="Semgrep config (auto, p/python, p/security, etc.)",
        ),
    ] = "auto",
) -> str:
    """Static analysis with semgrep."""
    project_root = get_project_root()
    cmd = ["uvx", "semgrep", "--config", config] + files
    result = run_command(cmd, cwd=project_root)
    return format_result(cmd, result)


@mcp.tool()
def deptry(
    exclude: Annotated[
        list[str], Field(default_factory=list, description="Patterns to exclude")
    ],
    root: Annotated[
        str, Field(default=".", description="Root directory to check")
    ] = ".",
    ignore_notebooks: Annotated[
        bool, Field(default=True, description="Ignore Jupyter notebooks")
    ] = True,
    ignore_missing: Annotated[
        bool, Field(default=False, description="Ignore missing dependencies")
    ] = False,
) -> str:
    """Find unused dependencies with deptry."""
    project_root = get_project_root()

    cmd = ["uvx", "--with", "deptry", "deptry", root]

    if ignore_notebooks:
        cmd.append("--ignore-notebooks")

    if ignore_missing:
        cmd.append("--ignore-missing")

    for pattern in exclude:
        cmd.extend(["--exclude", pattern])

    result = run_command(cmd, cwd=project_root)
    # Note that deptry returns non-zero exit codes when it finds issues,
    # which doesn't necessarily indicate a tool failure
    formatted_result = format_result(cmd, result)
    if result.returncode != 0 and "dependency issues" in result.stderr + result.stdout:
        return formatted_result.replace(
            "❌ Command failed", "⚠️ Dependencies issues found"
        )

    return formatted_result


def run_server():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run_server()
