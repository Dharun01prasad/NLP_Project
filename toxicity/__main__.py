import argparse
import json
from pathlib import Path

from .data import fetch_research_dataset
from .model import train_experiment


def main():
    parser = argparse.ArgumentParser(description="Mozhi training CLI")
    parser.add_argument("command", choices=["fetch", "train", "audit"])
    parser.add_argument("--runtime", type=Path, default=Path(__file__).resolve().parents[1] / "runtime")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    # Import lazily; normal module import does not initialize a runtime.
    from .server import Store
    store = Store(args.runtime)
    if args.command == "fetch":
        df, info = fetch_research_dataset(store.path / "data/source")
        print(json.dumps(store.install_data(df, info), ensure_ascii=False, indent=2))
    elif args.command == "train":
        result = train_experiment(store.dataframe(), store.read_json("data/info.json"),
                                  store.path / "models", store.path / "reports",
                                  augment=not args.no_augment, progress=lambda s: print(s, flush=True))
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    else:
        from .audit import audit_labels
        result = audit_labels(store.dataframe(), progress=lambda s: print(s, flush=True))
        result["revision"] = store.read_json("data/info.json")["revision"]
        store.write_json("reports/audit.json", result)
        print(f"Queued {len(result['rows'])} of {result['flagged_count']} flagged training examples")


if __name__ == "__main__":
    main()
