"""Validate the standalone notebook without importing Colab or using a GPU."""
import argparse
import ast
import json
from pathlib import Path
import re


def validate(path: Path, allow_placeholder: bool = False) -> dict[str, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    code_cells = 0
    pinned = False
    for number, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        assert cell["execution_count"] is None and cell["outputs"] == []
        source = "".join(cell["source"])
        tree = ast.parse(source, filename=f"{path.name}:cell-{number}")
        compile(tree, f"{path.name}:cell-{number}", "exec")
        for item in tree.body:
            if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SOURCE_COMMIT" for target in item.targets):
                value = ast.literal_eval(item.value)
                pinned = re.fullmatch(r"[0-9a-f]{40}", value) is not None
                if allow_placeholder and value == "PIN_SOURCE_COMMIT_BEFORE_RELEASE":
                    pinned = True
    assert pinned, "Notebook SOURCE_COMMIT must be pinned to its immutable implementation commit"
    assert code_cells >= 2
    return {"cells": len(notebook["cells"]), "compiled_code_cells": code_cells}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--allow-placeholder", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.notebook, args.allow_placeholder)))
