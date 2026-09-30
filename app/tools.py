from pathlib import Path
import subprocess
import sys


WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"


def safe_path(path: str) -> Path:
    """Resolve a path and make sure it stays inside workspace."""

    workspace = WORKSPACE.resolve()

    # Treat these as the workspace root.
    if path in ("", ".", "/", "\\"):
        return workspace

    target = (workspace / path).resolve()

    try:
        target.relative_to(workspace)
    except ValueError:
        raise ValueError("Access denied: path is outside the workspace.")

    return target


def list_files(path: str = ".") -> str:
    """List files and directories inside the workspace."""

    try:
        target = safe_path(path)
    except ValueError as exc:
        return str(exc)

    if not target.exists():
        return f"Path does not exist: {path}"

    if not target.is_dir():
        return f"Not a directory: {path}"

    items = []

    for item in sorted(target.iterdir()):
        kind = "DIR " if item.is_dir() else "FILE"
        items.append(f"{kind}  {item.name}")

    if not items:
        return "Directory is empty."

    return "\n".join(items)


def read_file(path: str) -> str:
    """Read a UTF-8 text file inside the workspace."""

    try:
        target = safe_path(path)
    except ValueError as exc:
        return str(exc)

    if not target.exists():
        return f"File does not exist: {path}"

    if not target.is_file():
        return f"Not a file: {path}"

    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Cannot read as UTF-8 text: {path}"
    except Exception as exc:
        return f"Error reading file: {exc}"


SAFE_COMMANDS = {
    "python_version": [sys.executable, "--version"],
    "git_version": ["git", "--version"],
    "git_status": ["git", "status", "--short", "--branch"],
    "git_log": ["git", "log", "-5", "--oneline"],
}


def safe_command(command: str) -> str:
    """
    Execute one predefined read-only command.

    Arbitrary shell commands are not accepted.
    The command always runs inside the workspace.
    """

    if command not in SAFE_COMMANDS:
        allowed = ", ".join(SAFE_COMMANDS.keys())
        return (
            f"Command '{command}' is not allowed. "
            f"Allowed commands: {allowed}"
        )

    args = SAFE_COMMANDS[command]

    try:
        result = subprocess.run(
            args,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out after 15 seconds."
    except FileNotFoundError as exc:
        return f"Command executable not found: {exc}"
    except Exception as exc:
        return f"Error running command: {exc}"

    output = result.stdout.strip()

    if result.stderr.strip():
        if output:
            output += "\n"
        output += result.stderr.strip()

    if not output:
        output = "(command returned no output)"

    return f"Exit code: {result.returncode}\n{output}"