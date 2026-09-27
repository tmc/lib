#!/usr/bin/env python3
"""Regenerate the checked-in SOLAR analyzer and performance fixtures."""

import json
import sys
import types
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: generate_solar_fixtures.py SOLAR_CHECKOUT FIXTURE_DIR...")

    solar_root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(solar_root))

    # SOLAR imports torchview while loading the graph analyzer. These fixtures
    # provide explicit graph data and do not execute torchview.
    sys.modules.setdefault("torchview", types.ModuleType("torchview"))

    from solar.analysis.graph_analyzer import EinsumGraphAnalyzer
    from solar.perf.perf_model import EinsumGraphPerfModel
    import yaml

    for fixture_arg in sys.argv[2:]:
        fixture = Path(fixture_arg).resolve()
        graph = fixture / "einsum_graph.yaml"
        arch = fixture / "configs" / "arch" / "FixtureArch.yaml"
        analysis_path = fixture / "analysis.yaml"

        analysis = EinsumGraphAnalyzer().analyze_graph(
            graph, fixture, precision="fp16", copy_graph=False
        )
        if analysis is None:
            raise SystemExit(f"SOLAR failed to analyze {graph}")
        perf = EinsumGraphPerfModel().predict(
            analysis_path,
            fixture,
            arch_config=arch,
            precision="fp16",
            copy_analysis=False,
        )
        if perf is None:
            raise SystemExit(f"SOLAR failed to predict {analysis_path}")

        for path in (analysis_path, fixture / "perf_FixtureArch.yaml"):
            value = yaml.safe_load(path.read_text())
            metadata = value.get("metadata") or {}
            if "source_graph" in metadata:
                metadata["source_graph"] = "einsum_graph.yaml"
                value["metadata"] = metadata
            path.write_text(json.dumps(value, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
