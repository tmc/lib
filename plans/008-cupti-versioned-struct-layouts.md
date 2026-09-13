# CUPTI versioned parameter and record layouts

Status: proposed  
Scope: `libgen`, `nvidia/cupti`, and `nvidia/x`  
Audience: maintainers of generated NVIDIA bindings and their runtime wrappers

## Purpose

CUPTI uses a leading `size` or `structSize` field to tell a native entry point
which portion of a parameter struct the caller supplies. The value is an ABI
boundary, not a payload length. A Go binding must therefore distinguish three
cases:

1. a layout whose complete `sizeof` is verified across every admitted runtime;
2. a layout whose header grows and must be passed as a version-specific prefix;
3. a record layout selected by the record kind and record version.

The binding must never use the newest generated `sizeof` as a substitute for a
runtime-specific prefix. That can make an older library read or write fields
that do not exist in its ABI.

## Evidence baseline

The qualification matrix currently covers these CUPTI API encodings:

| toolkit | reported API version |
| --- | ---: |
| CUDA 13.0 | 130001 |
| CUDA 13.1 | 130101 |
| CUDA 13.2 | 130202 |
| CUDA 13.3 | 130301 |
| CUDA 13.4 | 130400 |

The following `cupti_pcsampling.h` parameter layouts are byte-identical in all
five toolkits and may use complete `sizeof` initialization:

| struct | size |
| --- | ---: |
| `CUpti_PCSamplingConfigurationInfoParams` | 32 |
| `CUpti_PCSamplingGetDataParams` | 32 |
| `CUpti_PCSamplingEnableParams` | 24 |
| `CUpti_PCSamplingDisableParams` | 24 |
| `CUpti_PCSamplingStartParams` | 24 |
| `CUpti_PCSamplingStopParams` | 24 |
| `CUpti_PCSamplingGetNumStallReasonsParams` | 32 |
| `CUpti_PCSamplingGetStallReasonsParams` | 48 |

`CUpti_PCSamplingData` is a record and is 80 bytes. It is not a parameter
struct and must be handled by the record decoder.

The following parameter layouts are versioned prefixes:

| struct | runtime range | valid prefix end |
| --- | --- | --- |
| `CUpti_Profiler_Host_Initialize_Params` | 130001–130101 | `PHostObject` (48 bytes) |
| `CUpti_Profiler_Host_Initialize_Params` | 130202+ | `PSinglePassMetricSetName` (56 bytes) |
| `CUpti_Profiler_Host_GetMetricProperties_Params` | 130001 | `MetricType` (48 bytes) |
| `CUpti_Profiler_Host_GetMetricProperties_Params` | 130101+ | `MetricCollectionScope` (64 bytes) |

The apparent Go `sizeof` may include trailing alignment padding. Prefix sizes
must be computed as the end of the last supported field:

```go
uint(unsafe.Offsetof(p.PHostObject) + unsafe.Sizeof(p.PHostObject))
```

Activity records have a separate evolution policy. For example,
`CUpti_ActivityKernel` is 216 bytes (`Kernel10`) in CUDA 13.0 and 224 bytes
(`Kernel11`) from CUDA 13.1 onward. A parameter-size selector must never be
reused for activity record decoding.

## Generation contract

### Canonical generated types

Each generation run uses one pinned header closure, architecture, compiler, and
generator revision. The generated Go type is C-equivalent to that closure. The
generator emits layout tests for every generated struct and records the input
provenance. Regenerating from a newer closure may add fields to an existing Go
type; that is an ABI change that must be reviewed against the runtime matrix.

The generator does not silently manufacture compatibility variants from one
header. A single `ProfilerHostInitializeParams` therefore represents the
closure it was generated from, usually the newest admitted toolkit.

### Stable complete-size initialization

`APIConfig` may opt into zero-value size initialization only with an explicit
allowlist:

```go
AutoInitParamSizes      bool
AutoInitParamSizeStructs map[string]bool
```

For an allowlisted pointer-to-struct parameter, the generated wrapper emits:

```go
if p != nil && p.Size == 0 {
	p.Size = uint(unsafe.Sizeof(*p))
}
```

An explicit non-zero value is preserved. A first field named `size` or
`structSize` is not sufficient to qualify a struct. The allowlist is the
generator's evidence gate and must contain only layouts proven stable across
the admitted runtime set.

### Versioned prefix metadata

The generator should grow a declarative layout table for structs whose fields
were appended in later releases:

```go
type ParamLayout struct {
	MinVersion uint32
	MaxVersion uint32 // zero means no upper bound
	EndField   string
}

VersionedParamLayouts: map[string][]ParamLayout{
	"ProfilerHostInitializeParams": {
		{MinVersion: 130001, MaxVersion: 130201, EndField: "PHostObject"},
		{MinVersion: 130202, EndField: "PSinglePassMetricSetName"},
	},
}
```

From this metadata, generation should emit a selector or setter whose size is
derived from the named field. The generated code must reject a version with no
matching row and must reject an unknown field during generation. It must not
use `unsafe.Sizeof(struct)` for a versioned type.

One acceptable generated API is:

```go
func (p *ProfilerHostInitializeParams) SetStructSize(version uint32) error
```

The setter returns an error for an unsupported version and leaves the existing
value unchanged on failure. A wrapper that accepts a versioned parameter must
require the caller to qualify the runtime version before calling native code.

Generated selectors must also expose the selected end field or size in a
machine-readable layout table so tests and higher-level packages do not repeat
the ranges by hand.

### Records

Record generation must retain distinct Go types for distinct record kinds or
versions (`ActivityKernel10`, `ActivityKernel11`, and so on). Record decoders
must validate the native record header and available byte count before reading
fields. They may select a decoder from a runtime capability table, but they may
not infer a record's size from a parameter struct or from the current Go type's
trailing padding.

## Runtime and `nvidia/x` contract

The binding layer owns pointer conversion, pinning, bounded allocation, and
native error propagation. The `nvidia/x` layer owns runtime policy:

- obtain and normalize the library's reported version;
- qualify that version against generated layout metadata;
- choose the prefix row for versioned parameters;
- reject unsupported or ambiguous versions before a native call;
- expose domain-level capabilities and results.

`nvidia/x` must not duplicate numeric layout constants. Its code should call a
generated or binding-level selector such as `SetStructSize` or
`ParamLayoutFor`. Stable PC-sampling queries may use the binding's safe helper;
their reason names and lookup policy remain in `x/cuptipcsampling`.

Version normalization must preserve CUPTI's legacy CUDA 12 encoding. A decoded
major number alone is not qualification: the selected layout row must have
been measured for that runtime generation. A new point release may be admitted
only after the header/layout probe demonstrates that the relevant prefix is
unchanged.

## Failure rules

- nil parameter pointers return an error before native dispatch when the API
  cannot accept nil;
- zero size is filled only for an allowlisted stable layout;
- explicit non-zero sizes are never overwritten;
- unknown runtime versions fail closed;
- a versioned parameter with no matching prefix row fails before dispatch;
- a generated field name missing from the current struct fails generation;
- a record with an unknown kind, short payload, or incompatible version fails
  decoding without reading beyond the supplied bytes;
- no source-only layout result is reported as runtime compatibility.

## Required tests

### Generator tests

1. An allowlisted stable struct receives zero-size initialization.
2. A struct with a leading size field but no allowlist entry does not.
3. A versioned struct never receives complete `sizeof` initialization.
4. Every generated `EndField` resolves and produces compilable Go.
5. Explicit non-zero sizes remain unchanged in generated wrappers.

### Layout tests

For every admitted architecture and toolkit closure, generate tests that pin:

- `unsafe.Sizeof` and alignment for complete stable structs;
- `unsafe.Offsetof(lastField) + unsafe.Sizeof(lastField)` for each prefix row;
- record sizes and field offsets for each activity record version.

### Runtime probes

Run a bounded native probe against each installed CUPTI toolkit. Record the
reported version, header hashes, selected layout row, symbol resolution, and
cleanup result. Include negative probes for an unknown version and an
intentionally oversized prefix. A successful symbol lookup or struct compile
alone is not runtime qualification.

## Migration order

1. Keep the current stable PC-sampling allowlist and layout tests.
2. Add generated versioned-prefix metadata and selector APIs.
3. Migrate `x/cuptirange` from hand-written field-end arithmetic to the
   generated selectors, retaining its runtime qualification policy.
4. Add activity record version metadata and bounded decoders.
5. Regenerate each package only from a pinned, reproducible header closure;
   never hand-edit generated output.

The acceptance condition is that every supported call has either a verified
complete layout or an explicitly selected versioned prefix, with no duplicated
magic sizes in higher-level packages and no claim of compatibility for an
untested runtime.
