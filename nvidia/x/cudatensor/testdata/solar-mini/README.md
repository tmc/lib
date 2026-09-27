This fixture's graph was passed through SOLAR's `EinsumGraphAnalyzer` and
`EinsumGraphPerfModel` on 2026-09-27. The `FixtureArch` values are deliberately
small deterministic rates, not hardware measurements. The graph and generated
analysis/performance documents are stored as JSON, which is valid YAML 1.2 and
lets Go tests decode them with the standard library. The architecture document
is stored in the same form under `configs/arch/`.

The test environment lacks SOLAR's optional `torchview` dependency. Fixture
generation imported a no-op module stub for `torchview`; neither analyzer nor
performance modeling calls it. No Python process is used by tests.
