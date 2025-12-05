import subprocess
import argparse
import json
import ast
from pathlib import Path
from clearml import Dataset

import libcst as cst


def clone_repo(repo_url: str, output_dir: Path) -> Path:
    """
    Clone the given git repository into output_dir. Returns the path to the cloned repo.
    If the destination exists, skips cloning.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_name = Path(repo_url.rstrip("/{}").split("/")[-1]).stem
    dest = output_dir / repo_name
    if dest.exists():
        print(f"Directory {dest} already exists, skipping clone.")
    else:
        subprocess.run(["git", "clone", repo_url, str(dest)], check=True)
    return dest


def extract_functions(repo_dir: Path) -> list:
    """
    Recursively parse all .py files under repo_dir using LibCST and extract functions.
    Returns a list of dicts: {name, signature, docstring, code}.
    """
    functions = []
    for py_file in repo_dir.rglob("*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            module = cst.parse_module(source)
        except (cst.ParserSyntaxError, UnicodeDecodeError):
            continue

        collector = FunctionCollector()
        module.visit(collector)

        for node in collector.functions:
            code = module.code_for_node(node)
            sig = code.split(":", 1)[0] + ":"
            doc = ""
            if isinstance(node.body, cst.IndentedBlock) and node.body.body:
                first = node.body.body[0]
                if isinstance(first, cst.SimpleStatementLine):
                    stmt = first.body
                    if (
                        stmt
                        and isinstance(stmt[0], cst.Expr)
                        and isinstance(stmt[0].value, cst.SimpleString)
                    ):
                        literal = stmt[0].value.value
                        try:
                            doc = ast.literal_eval(literal)
                        except Exception:
                            doc = literal.strip('"')
            functions.append(
                {
                    "name": node.name.value,
                    "signature": sig,
                    "docstring": doc,
                    "code": code,
                }
            )
    return functions


def organize_functions(functions: list, output_dir: Path):
    """
    Write each function to functions/<name>.py and generate dataset.jsonl
    where each line is {"prompt":..., "completion":...} for fine-tuning.
    """
    funcs_dir = output_dir / "functions"
    funcs_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "dataset.jsonl"

    with dataset_path.open("w", encoding="utf-8") as ds:
        for func in functions:
            file_path = funcs_dir / f"{func['name']}.py"
            file_path.write_text(func["code"], encoding="utf-8")

            prompt = func["signature"]
            if func["docstring"]:
                prompt += '\n"""' + func["docstring"] + '"""'
            prompt += "\n"
            completion = func["code"]

            entry = {"prompt": prompt, "completion": completion}
            ds.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {len(functions)} functions into {funcs_dir}")
    print(f"Generated dataset at {dataset_path}")


class FunctionCollector(cst.CSTVisitor):
    """
    Collect all function definitions in the module.
    """

    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.functions.append(node)


def main():
    parser = argparse.ArgumentParser(
        description="Clone a repo, extract Python functions with LibCST, and prepare them for LLM fine-tuning."
    )
    parser.add_argument("repo_url", help="Git repository URL to clone")
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Directory to store cloned repo, functions, and dataset",
    )
    parser.add_argument(
        "--dataset-name",
        default="Finetune Example",
        help="The name of the ClearML dataset to dump the data to",
    )
    parser.add_argument(
        "--dataset-project",
        default="Finetune Example",
        help="The name of the ClearML dataset project to dump the data to",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    repo_dir = clone_repo(args.repo_url, out_dir)
    funcs = extract_functions(repo_dir)
    organize_functions(funcs, out_dir)
    dataset = Dataset.create(
        dataset_name=args.dataset_name, dataset_project=args.dataset_project
    )
    dataset.add_files(args.output, wildcard="*.jsonl")
    dataset.finalize(auto_upload=True)


if __name__ == "__main__":
    main()
