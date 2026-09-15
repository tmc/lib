# Compare external measurements with Go benchmarks

`cuptibench-import` converts explicit measurements from C or another producer
into the same text format as Go benchmarks. It does not run the C program,
infer its timing boundaries, or parse an unspecified C tool's output.

From the `nvidia` module, exercise the **synthetic parser fixtures**:

```sh
go run ./cmd/cuptibench-import ./x/cuptibench/benchfmt/testdata/synthetic-c.json > ~/tmp/cuptibench-c.txt
benchstat ~/tmp/cuptibench-c.txt ./x/cuptibench/benchfmt/testdata/synthetic-go.txt
```

Both files contain invented numbers for checking the format. These commands
produce one comparison row with one sample per arm. Benchstat reports that
there are too few samples for a confidence interval or a significant difference.

## Canonical input

Supply one JSON document with `version: 1`, `config`, and `samples`. See
[testdata/synthetic-c.json](testdata/synthetic-c.json) for a complete fixture.
Each sample records:

- `name`: the **exact** Go benchmark name, including its subbenchmark path and
  processor suffix if present. For example, `BenchmarkDecode/full-8`. The `-8`
  matches the Go row; it does not claim that the C producer used eight threads.
  The converter never appends `-C` or invents a processor count.
- `sample_id` and `source`: identify the original measurement and its artifact.
- `operations`: the number of logical operations represented by that sample.
- `scope`: `full`, `isolated`, or `attributed`.
- `provenance`: source-specific facts such as artifact hash, executable revision,
  model/input hashes, preparation, stream policy, and timing boundaries.
- `measurements`: explicit values, units, kinds, and normalization bases.

Copy the comparison arm's `goos`, `goarch`, `pkg`, and `cpu` headers into `config`
where appropriate. Benchstat groups differences in these headers into separate
tables. Sample provenance is emitted as JSON comments, which retains it without
splitting comparison rows by C/Go implementation or run identity.

## Normalization and timing

| Kind | Output unit | Permitted basis |
| --- | --- | --- |
| `host-elapsed` | `ns/op` | `total`, `per-operation` |
| `device-elapsed` | `gpu-ns/op` | `total`, `per-operation` |
| `kernel-duration-sum` | `kernel-sum-ns/op` | `total`, `per-operation` |
| `counter` | Exact unit; `/op` when normalized | `raw`, `total`, `per-operation` |
| `rate`, `ratio` | Exact source unit | `raw` |

`total` divides the source value by `operations`. `per-operation` and `raw`
preserve the source value. `ns/op` specifically means the matching Go host
timing boundary (including the same submission and completion waits); the
converter does not rename an unrelated wall-clock interval to make it compare. Values use float64 precision. A missing value is an
error. A measured counter may be zero; a zero elapsed duration is unresolved
and rejected.

When comparing cuptibench plans with `Config.Operations > 1`, Go host `ns/op`
is per benchmark sequence and `gpu-ns/op` is per logical operation. Match these
denominators explicitly. The importer cannot infer a Go plan from a benchmark
name or normalize the other comparison file.

One source sample produces **one row**, even if `operations` is 128. It does not
produce 128 independent statistical samples. To compare repeated runs, retain
independent source measurements and give each its own sample ID. Duplicate
source/sample/name combinations are rejected.

Match workload scopes and operation definitions before comparing values. A sum
of kernel durations is not whole-pass latency when kernels overlap. A host
interval is not a CUDA-event interval. Raw CUPTI sequence totals must represent
the same sequence size on both arms. The converter enforces declared kinds and
normalization; it cannot verify those declarations against an unknown C tool.

## CLI

```sh
go run ./cmd/cuptibench-import retained-run.json > ~/tmp/cuptibench-c.txt
# Or provide the JSON on standard input:
go run ./cmd/cuptibench-import < retained-run.json
```

JSON input is limited to 64 MiB, including trailing whitespace, while reading.
Invalid or oversized input produces no benchmark text and a nonzero exit status. The format
uses standard library Go code and does not require CUDA or CUPTI libraries.
