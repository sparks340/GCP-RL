#!/usr/bin/env python3
"""一键环境检查脚本：校验项目关键依赖并输出简要运行建议。"""

import importlib
import platform
import sys
from typing import List, Tuple

DEPENDENCIES: List[Tuple[str, str]] = [
    ("numpy", "numpy"),
    ("networkx", "networkx"),
    ("gymnasium", "gymnasium"),
    ("torch", "torch"),
    ("tianshou", "tianshou"),
    ("torch_geometric", "torch-geometric"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
    ("tensorboard", "tensorboard"),
]


def get_version(module) -> str:
    for attr in ("__version__", "VERSION"):
        if hasattr(module, attr):
            value = getattr(module, attr)
            if isinstance(value, str):
                return value
            if isinstance(value, tuple):
                return ".".join(str(v) for v in value)
    return "unknown"


def main() -> int:
    print("=== GCP-RL 环境检查 ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    failed = []
    for module_name, pip_name in DEPENDENCIES:
        try:
            module = importlib.import_module(module_name)
            version = get_version(module)
            print(f"[OK] {pip_name:<15} version={version}")
        except Exception as exc:  # noqa: BLE001
            failed.append((pip_name, str(exc)))
            print(f"[MISSING] {pip_name:<15} reason={exc}")

    if failed:
        print("\n环境检查结果：存在缺失依赖。")
        print("建议执行：")
        print("  pip install -r requirements.txt")
        print("\n缺失列表：")
        for pip_name, reason in failed:
            print(f"  - {pip_name}: {reason}")
        return 1

    print("\n环境检查结果：通过 ✅")
    print("你可以继续运行：")
    print("  python trainer.py <output_policy.pth>")
    print("  python runner.py test_graph.txt --ablation no_rl --search-algorithm tabu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
