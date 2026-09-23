package main

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

// PTXConfig specifies assembly emission parameters.
type PTXConfig struct {
	Version     string
	Target      string
	AddressSize int
}

// DefaultPTXConfig returns default configuration for sm_90 with PTX 7.8.
func DefaultPTXConfig() PTXConfig {
	return PTXConfig{
		Version:     "7.8",
		Target:      "sm_90",
		AddressSize: 64,
	}
}

// ModulePTX returns a full PTX module containing both producer_subgrid and
// consumer_subgrid with programmatic dependent launch (PDL) synchronization.
func ModulePTX(cfg PTXConfig) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("// PTX ISA %s, target %s\n", cfg.Version, cfg.Target))
	b.WriteString(fmt.Sprintf(".version %s\n", cfg.Version))
	b.WriteString(fmt.Sprintf(".target %s\n", cfg.Target))
	b.WriteString(fmt.Sprintf(".address_size %d\n\n", cfg.AddressSize))

	b.WriteString(ProducerPTX(cfg))
	b.WriteString("\n")
	b.WriteString(ConsumerPTX(cfg))
	return b.String()
}

// ProducerPTX returns the PTX assembly for producer_subgrid.
// It computes tiles up to trigger_tile, issues a GPU-scope release fence,
// executes griddepcontrol.launch_dependents, and continues computing
// subsequent tiles.
func ProducerPTX(cfg PTXConfig) string {
	return `.visible .entry producer_subgrid(
	.param .u64 tiles_out,
	.param .u32 num_tiles,
	.param .u32 tile_size,
	.param .u32 trigger_tile
)
{
	.reg .b64 %out_ptr, %rd<8>;
	.reg .b32 %ntiles, %tsize, %trig, %t, %tid, %ctaid, %r<10>;
	.reg .pred %p<6>;

	ld.param.u64 %out_ptr, [tiles_out];
	ld.param.u32 %ntiles, [num_tiles];
	ld.param.u32 %tsize, [tile_size];
	ld.param.u32 %trig, [trigger_tile];

	mov.u32 %tid, %tid.x;
	mov.u32 %ctaid, %ctaid.x;

	// Early tile production loop (tiles 0 to trigger_tile - 1)
	mov.u32 %t, 0;
$early_loop:
	setp.ge.u32 %p1, %t, %trig;
	@%p1 bra $early_done;

	// Work on tile %t
	mul.wide.u32 %rd1, %t, %tsize;
	add.s64 %rd2, %out_ptr, %rd1;
	st.global.u32 [%rd2], %t;

	add.u32 %t, %t, 1;
	bra $early_loop;

$early_done:
	// Release fence synchronizes with dependent grid start
	fence.sc.gpu;

	// Programmatic Dependent Launch: signal that downstream dependents
	// may be scheduled once all CTAs in this grid reach this instruction.
	griddepcontrol.launch_dependents;

	// Subsequent tile production loop (tiles trigger_tile to num_tiles - 1)
$late_loop:
	setp.ge.u32 %p2, %t, %ntiles;
	@%p2 bra $late_done;

	// Work on subsequent tile %t
	mul.wide.u32 %rd3, %t, %tsize;
	add.s64 %rd4, %out_ptr, %rd3;
	st.global.u32 [%rd4], %t;

	add.u32 %t, %t, 1;
	bra $late_loop;

$late_done:
	ret;
}
`
}

// ConsumerPTX returns the PTX assembly for consumer_subgrid.
// It immediately processes early tiles (tiles 0 to trigger_tile - 1),
// then waits for the prerequisite grid to complete via griddepcontrol.wait
// before processing remaining tiles.
func ConsumerPTX(cfg PTXConfig) string {
	return `.visible .entry consumer_subgrid(
	.param .u64 tiles_in,
	.param .u64 result_out,
	.param .u32 num_tiles,
	.param .u32 tile_size,
	.param .u32 trigger_tile
)
{
	.reg .b64 %in_ptr, %out_ptr, %rd<8>;
	.reg .b32 %ntiles, %tsize, %trig, %t, %tid, %ctaid, %val, %r<10>;
	.reg .pred %p<6>;

	ld.param.u64 %in_ptr, [tiles_in];
	ld.param.u64 %out_ptr, [result_out];
	ld.param.u32 %ntiles, [num_tiles];
	ld.param.u32 %tsize, [tile_size];
	ld.param.u32 %trig, [trigger_tile];

	mov.u32 %tid, %tid.x;
	mov.u32 %ctaid, %ctaid.x;

	// Early consumption: tiles 0 to trigger_tile - 1 are already visible
	mov.u32 %t, 0;
$cons_early_loop:
	setp.ge.u32 %p1, %t, %trig;
	@%p1 bra $cons_early_done;

	mul.wide.u32 %rd1, %t, %tsize;
	add.s64 %rd2, %in_ptr, %rd1;
	ld.global.u32 %val, [%rd2];
	add.s64 %rd3, %out_ptr, %rd1;
	st.global.u32 [%rd3], %val;

	add.u32 %t, %t, 1;
	bra $cons_early_loop;

$cons_early_done:
	// Wait until prerequisite producer grid completes and all memory
	// operations are visible to this grid.
	griddepcontrol.wait;

	// Subsequent consumption: tiles trigger_tile to num_tiles - 1
$cons_late_loop:
	setp.ge.u32 %p2, %t, %ntiles;
	@%p2 bra $cons_late_done;

	mul.wide.u32 %rd4, %t, %tsize;
	add.s64 %rd5, %in_ptr, %rd4;
	ld.global.u32 %val, [%rd5];
	add.s64 %rd6, %out_ptr, %rd4;
	st.global.u32 [%rd6], %val;

	add.u32 %t, %t, 1;
	bra $cons_late_loop;

$cons_late_done:
	ret;
}
`
}

// SequentialModulePTX returns a standard sequential module without griddepcontrol.
func SequentialModulePTX(cfg PTXConfig) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("// PTX ISA %s, target %s (Sequential)\n", cfg.Version, cfg.Target))
	b.WriteString(fmt.Sprintf(".version %s\n", cfg.Version))
	b.WriteString(fmt.Sprintf(".target %s\n", cfg.Target))
	b.WriteString(fmt.Sprintf(".address_size %d\n\n", cfg.AddressSize))

	b.WriteString(`.visible .entry producer_sequential(
	.param .u64 tiles_out,
	.param .u32 num_tiles,
	.param .u32 tile_size
)
{
	.reg .b64 %out_ptr, %rd<6>;
	.reg .b32 %ntiles, %tsize, %t, %r<6>;
	.reg .pred %p<2>;

	ld.param.u64 %out_ptr, [tiles_out];
	ld.param.u32 %ntiles, [num_tiles];
	ld.param.u32 %tsize, [tile_size];

	mov.u32 %t, 0;
$seq_loop:
	setp.ge.u32 %p1, %t, %ntiles;
	@%p1 bra $seq_done;

	mul.wide.u32 %rd1, %t, %tsize;
	add.s64 %rd2, %out_ptr, %rd1;
	st.global.u32 [%rd2], %t;

	add.u32 %t, %t, 1;
	bra $seq_loop;

$seq_done:
	ret;
}

.visible .entry consumer_sequential(
	.param .u64 tiles_in,
	.param .u64 result_out,
	.param .u32 num_tiles,
	.param .u32 tile_size
)
{
	.reg .b64 %in_ptr, %out_ptr, %rd<8>;
	.reg .b32 %ntiles, %tsize, %t, %val, %r<8>;
	.reg .pred %p<2>;

	ld.param.u64 %in_ptr, [tiles_in];
	ld.param.u64 %out_ptr, [result_out];
	ld.param.u32 %ntiles, [num_tiles];
	ld.param.u32 %tsize, [tile_size];

	mov.u32 %t, 0;
$seq_cons_loop:
	setp.ge.u32 %p1, %t, %ntiles;
	@%p1 bra $seq_cons_done;

	mul.wide.u32 %rd1, %t, %tsize;
	add.s64 %rd2, %in_ptr, %rd1;
	ld.global.u32 %val, [%rd2];
	add.s64 %rd3, %out_ptr, %rd1;
	st.global.u32 [%rd3], %val;

	add.u32 %t, %t, 1;
	bra $seq_cons_loop;

$seq_cons_done:
	ret;
}
`)
	return b.String()
}

// AuditModule parses the PTX text and validates instruction opcodes and
// architecture compatibility.
func AuditModule(ptxText string) (*cudaptx.Module, error) {
	m, err := cudaptx.Parse(strings.NewReader(ptxText))
	if err != nil {
		return nil, fmt.Errorf("audit: parse ptx: %w", err)
	}

	// Verify required opcodes for producer
	if err := m.AssertOpcodes("producer_subgrid", []string{
		"griddepcontrol.launch_dependents",
		"fence.sc.gpu",
	}, nil); err != nil {
		return nil, fmt.Errorf("audit: producer opcodes: %w", err)
	}

	// Verify required opcode for consumer
	if err := m.AssertOpcodes("consumer_subgrid", []string{
		"griddepcontrol.wait",
	}, nil); err != nil {
		return nil, fmt.Errorf("audit: consumer opcodes: %w", err)
	}

	// Verify architecture compatibility (griddepcontrol requires sm_90+)
	if err := m.AuditArchitecture(""); err != nil {
		return nil, fmt.Errorf("audit: architecture: %w", err)
	}

	return m, nil
}
