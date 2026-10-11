# Operation comparison summary

`comparison-summary` combines existing receipts; it does not recollect or
requalify a capture. First save shape output and extract the alignment receipt
from the comparison TAR:

```sh
cudaprof shape-compare base.pprof target.pprof > "$HOME/tmp/shape.json"
tar -xf comparison.tar -C "$HOME/tmp/comparison"
cudaprof comparison-summary -shape "$HOME/tmp/shape.json" \
  -alignment "$HOME/tmp/comparison/.alignment.json" \
  -o "$HOME/tmp/operation-summary"
```

The output directory must be new. Open `summary.html` directly; it needs no
server or external scripts. `summary.json` retains the input receipts and raw
capture identities. An optional `-campaign campaign.json` displays the existing
runner's independent block interval and excluded blocks without recomputing it.

Capture SHA identities must agree between the shape and alignment receipts.
Time stretching is refused. Completeness, expected marker coverage, selected
stationarity, full capture stationarity and shape qualification stay separate.
Only a qualified shape receipt with complete matching identities produces
numerical equivalent-operation comparisons. Unmatched kernels and missing
fields remain visible. Old unmarked captures produce an unqualified summary,
not invented operation labels.

Operation distributions cover the full decoded capture used by shape-compare;
they do not silently adopt the alignment receipt's selected window. Kernel
sums can overlap and are separate from elapsed windows and campaign delivery.
Launch counts are not independent samples. Campaign intervals retain their
original conditional exchangeability assumption.

Each matched launch carries capture SHA, process-generation journal, original
decoded activity index, raw symbol and original/aligned timestamp. These are
stable references within the named capture; the index is not a Perfetto slice
ID. In the exported timeline locate the journal/raw-symbol annotations at the
aligned timestamp. The view names both trace files and retains exact fixed
offsets without synchronization or stretching. Copy trace files alongside the
receipts when sharing the summary.

CPU validation:

```sh
cd nvidia
go test ./x/cmd/cudaprof -run '^TestComparisonSummary' -count=1
```

Tests cover unavailable application fields, differing counts, failed gates,
identity mismatch, stretched time, retained campaign exclusions and HTML
escaping. A fresh GPU workload demonstration remains separate validation.

On existing Python/Go Gemma EOS captures, CPU-only summary generation retained
passed collector and expected-marker gates alongside failed stationarity and
shape qualification. It emitted zero equivalent numerical operation rows,
with 11,388 Python and 11,908 Go kernels unmatched, and time scale 1. The
exact local validation artifacts are machine-local and not committed;
this diagnostic result does not qualify an operation slowdown.
