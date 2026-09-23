# Sub-grid early release

This example generates a producer and consumer kernel pair that overlap
through Programmatic Dependent Launch (PDL) and the PTX instruction
`griddepcontrol`, audits the generated PTX, and evaluates an analytic cycle
model of the overlap. Nothing runs on a GPU. PDL needs `sm_90` or later.

## griddepcontrol and PDL

```ptx
griddepcontrol.action;

.action = { .launch_dependents, .wait }
```

- `griddepcontrol.launch_dependents` lets the grids that depend on this one
  be scheduled once every CTA of this grid has issued it or completed. The
  dependent grid may launch before this one finishes. Repeating it within a
  CTA has no further effect.
- `griddepcontrol.wait` waits until all prerequisite grids have completed
  and their memory operations are visible to this grid. A consumer of a
  grid that releases early must wait before reading what the producer
  writes after the release.

A release fence before `griddepcontrol.launch_dependents` synchronizes with
the start of the dependent grid if the fence has `sys` scope, or if it has
`gpu` scope and both grids share a memory synchronization domain. The
producer therefore issues

```ptx
fence.sc.gpu;
griddepcontrol.launch_dependents;
```

before computing its remaining tiles.

On the host, PDL is enabled per launch with the
`driver.LaunchAttributeProgrammaticStreamSerialization` launch attribute,
which `x/cudalaunch` sets with:

```go
l := cudalaunch.Kernel(fn).
	Grid(numBlocks).
	Block(threadsPerBlock).
	WithProgrammaticStreamSerialization(true)
```

## Kernels

The producer, `producer_subgrid`, divides each CTA's work into tiles. It
computes the tiles before the trigger, issues the fence and
`griddepcontrol.launch_dependents`, and then computes the rest while the
consumer runs.

The consumer, `consumer_subgrid`, can be scheduled once every producer CTA
has reached the trigger. It processes the released tiles, executes
`griddepcontrol.wait`, which stalls only if the producer has not finished,
and then processes the remaining tiles.

## The cost model

`simulate.go` models three ways to run the pair: unpipelined sequential
launches, sequential CUDA graph nodes, and PDL with the trigger after 25%,
50%, and 75% of the tiles. It evaluates them at three grid sizes: fewer CTAs
than SMs, one CTA per SM, and two waves. No kernel is launched: every
duration is model cycles divided by an assumed clock, and the default
hardware is an assumed H100. The report says so in its header.

For each mode the report gives the modeled time, the time both kernels are
active (overlap), overlap as a share of the total, the modeled SM occupancy,
and the speedup against the sequential launches and against the graph.
`-format=md` adds an ASCII timeline of SM occupancy for each scenario.

The trade-off the model shows is the trigger point. An early trigger gives
the longest overlap but can leave the consumer stalled at
`griddepcontrol.wait`; a late one avoids the stall but overlaps less. None
of this is measured on a device.

## Usage

From this directory:

```sh
GOWORK=off go run .                # cost model as a table
GOWORK=off go run . -format=md     # markdown, with the occupancy timeline
GOWORK=off go run . -format=json
GOWORK=off go run . -sms 148       # assume another SM count
GOWORK=off go run . -ptx           # print the generated PTX
GOWORK=off go run . -audit         # check the PTX opcodes and target with x/cudaptx
GOWORK=off go test -bench=. .
```

`-audit` parses the PTX with `x/cudaptx`, checks that the producer issues
`fence.sc.gpu` and `griddepcontrol.launch_dependents` and the consumer
`griddepcontrol.wait`, and checks the target architecture supports them.
