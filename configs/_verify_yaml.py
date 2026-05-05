import os
import sys
import yaml


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        os.path.join(base, "ablation"),
        os.path.join(base, "comparison"),
    ]
    passed = 0
    failed = 0
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[SKIP] Directory not found: {d}")
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".yaml"):
                continue
            fpath = os.path.join(d, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    raise ValueError("Top-level is not a dict")
                expected_name = fname.replace(".yaml", "")
                actual_name = data.get("experiment_name")
                if actual_name != expected_name:
                    raise ValueError(
                        f"experiment_name mismatch: expected '{expected_name}', got '{actual_name}'"
                    )
                print(f"[PASS] {os.path.relpath(fpath, base)}")
                passed += 1
            except Exception as e:
                print(f"[FAIL] {os.path.relpath(fpath, base)}: {e}")
                failed += 1

    print(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
