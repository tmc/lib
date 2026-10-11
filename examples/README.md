# Cross-module tools

This unreleased module contains programs that span the repository’s modules,
including inference examples and benchmark tools. Local replace directives
resolve adjacent module checkouts. Run commands from this directory; commands
live under `x/cmd`.

```
go run ./x/cmd/profile2trace -spans host.spans.jsonl > host.pftrace
go run ./x/cmd/hostqualify -max-load 4
go run ./x/cmd/inferbenchhash model-directory
go run ./x/cmd/inferbenchcheck campaign.json raw-directory
```

`profile2trace` converts supplied host, Metal and Nsight traces to Perfetto.
Combining clocks requires explicit offsets; see its [README](x/cmd/profile2trace/README.md).
`hostqualify` checks observed host load and optional reference throughput.
It does not qualify a GPU, kernel, or the interval between observations.
`inferbenchcheck` validates matched benchmark receipts and their retained raw
set; declarations in a receipt do not independently prove physical execution.

`internal/inferbench` supplies matched campaign, provenance and lease helpers.
The package does not implement a model backend or acquire hardware.
