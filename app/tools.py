from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"


def safe_path(path: str) -> Path:
    """Resolve a path and make sure it stays inside workspace."""

    workspace = WORKSPACE.resolve()
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