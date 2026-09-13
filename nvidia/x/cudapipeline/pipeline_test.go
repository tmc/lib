package cudapipeline_test

import (
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func TestValidatePTX_Table(t *testing.T) {
	tests := []struct {
		name          string
		ptx           string
		opts          []cudapipeline.Option
		wantViolation string
		wantCount     int
	}{
		{
			name: "valid_2stage_tma_pipeline",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry valid_tma_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_tmap, [tmap];

    // Initialize barrier with 1 arrival token
    mbarrier.init.shared::cta.b64 [bar], 1;

    // Phase 0: expect 1024 bytes and arrive
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;

    // TMA copy transfers 1024 bytes
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];

    // Wait on phase 0 (parity 0)
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;

    // Phase 1: expect 1024 bytes and arrive
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;

    // TMA copy transfers 1024 bytes
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {16, 0}], [bar];

    // Wait on phase 1 (parity 1)
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 1;

    ret;
}
`,
			opts: []cudapipeline.Option{
				cudapipeline.WithTensorSize("tmap", 1024),
			},
			wantCount: 0,
		},
		{
			name: "uninitialized_barrier_arrive",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry uninit_barrier_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .reg .b64 %state;
    .shared .align 8 .b64 bar;

    // BUG: Missing mbarrier.init
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    ret;
}
`,
			wantViolation: cudapipeline.ErrUninitializedBarrier,
			wantCount:     1,
		},
		{
			name: "uninitialized_barrier_copy",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry uninit_copy_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .shared .align 8 .b64 bar;

    // BUG: mbarrier.init never called before TMA copy
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];
    ret;
}
`,
			wantViolation: cudapipeline.ErrUninitializedBarrier,
			wantCount:     1,
		},
		{
			name: "unbalanced_transaction_bytes_underflow",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry underflow_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_tmap, [tmap];
    mbarrier.init.shared::cta.b64 [bar], 1;

    // Expect 2048 bytes
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 2048;

    // Only copy 1024 bytes
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];

    // Wait: expected 2048 bytes, completed 1024 bytes
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`,
			opts: []cudapipeline.Option{
				cudapipeline.WithTensorSize("tmap", 1024),
			},
			wantViolation: cudapipeline.ErrMismatchedTxBytes,
			wantCount:     1,
		},
		{
			name: "unbalanced_transaction_bytes_no_copy",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry no_copy_kernel()
{
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;

    // BUG: No copy issued before wait
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`,
			wantViolation: cudapipeline.ErrMismatchedTxBytes,
			wantCount:     1,
		},
		{
			name: "unexpected_transaction_without_expect_tx",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry unexpected_tx_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .reg .b64 %state;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_tmap, [tmap];
    mbarrier.init.shared::cta.b64 [bar], 1;

    // Arrive without expect_tx
    mbarrier.arrive.shared::cta.b64 %state, [bar];

    // BUG: Copy issued without expect_tx on this barrier
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];
    ret;
}
`,
			wantViolation: cudapipeline.ErrUnexpectedTx,
			wantCount:     1,
		},
		{
			name: "phase_parity_error_untoggled",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry parity_error_kernel(
    .param .u64 tmap
)
{
    .reg .b64 %rd_tmap;
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    ld.param.u64 %rd_tmap, [tmap];
    mbarrier.init.shared::cta.b64 [bar], 1;

    // Phase 0
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {0, 0}], [bar];
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;

    // Phase 1
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%rd_tmap, {16, 0}], [bar];

    // BUG: Waiting on parity 0 again instead of parity 1
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`,
			opts: []cudapipeline.Option{
				cudapipeline.WithTensorSize("tmap", 1024),
			},
			wantViolation: cudapipeline.ErrPhaseParity,
			wantCount:     1,
		},
		{
			name: "invalid_parity_value",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry invalid_parity_kernel()
{
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    // Parity operand is 2 (must be 0 or 1)
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 2;
    ret;
}
`,
			wantViolation: cudapipeline.ErrInvalidParity,
			wantCount:     2, // invalid parity + missing arrive
		},
		{
			name: "missing_arrive_before_wait",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry missing_arrive_kernel()
{
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;

    // BUG: try_wait called without any prior arrive
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`,
			wantViolation: cudapipeline.ErrMissingArrive,
			wantCount:     1,
		},
		{
			name: "duplicate_barrier_init",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry dup_init_kernel()
{
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    // BUG: Re-initializing active barrier without mbarrier.inval
    mbarrier.init.shared::cta.b64 [bar], 1;
    ret;
}
`,
			wantViolation: cudapipeline.ErrDuplicateInit,
			wantCount:     1,
		},
		{
			name: "invalidated_barrier_use",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry inval_use_kernel()
{
    .reg .b64 %state;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.inval.shared::cta.b64 [bar];

    // BUG: Arriving on invalidated barrier
    mbarrier.arrive.shared::cta.b64 %state, [bar];
    ret;
}
`,
			wantViolation: cudapipeline.ErrInvalidatedBarrier,
			wantCount:     1,
		},
		{
			name: "valid_bulk_async_pipeline",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry bulk_async_kernel()
{
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 512;
    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [0], [0], 512, [bar];
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`,
			wantCount: 0,
		},
		{
			name: "unbalanced_bulk_async_pipeline",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry bulk_unbalanced_kernel()
{
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], 1024;
    // Only 512 bytes copied
    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [0], [0], 512, [bar];
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], 0;
    ret;
}
`,
			wantViolation: cudapipeline.ErrMismatchedTxBytes,
			wantCount:     1,
		},
		{
			name: "constant_register_resolution",
			ptx: `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry reg_resolve_kernel()
{
    .reg .u32 %r_bytes;
    .reg .u32 %r_par;
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar;

    mov.u32 %r_bytes, 256;
    mov.u32 %r_par, 0;

    mbarrier.init.shared::cta.b64 [bar], 1;
    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], %r_bytes;
    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [0], [0], 256, [bar];
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar], %r_par;
    ret;
}
`,
			wantCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mod, err := cudaptx.Parse(strings.NewReader(tt.ptx))
			if err != nil {
				t.Fatalf("Parse PTX: %v", err)
			}

			violations, err := cudapipeline.ValidatePTX(mod, tt.opts...)
			if err != nil {
				t.Fatalf("ValidatePTX: %v", err)
			}

			if tt.wantCount == 0 && len(violations) > 0 {
				t.Fatalf("expected 0 violations, got %d: %v", len(violations), violations)
			}
			if tt.wantCount > 0 && len(violations) != tt.wantCount {
				t.Fatalf("expected %d violations, got %d: %v", tt.wantCount, len(violations), violations)
			}

			if tt.wantViolation != "" {
				found := false
				for _, v := range violations {
					if v.Violation == tt.wantViolation {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("expected violation %q in %v", tt.wantViolation, violations)
				}
			}
		})
	}
}

func TestBarrierTracker_Barriers(t *testing.T) {
	ptx := `
.version 8.8
.target sm_121a
.address_size 64

.visible .entry tracker_inspect_kernel()
{
    .reg .b64 %state;
    .reg .pred %p0;
    .shared .align 8 .b64 bar0;
    .shared .align 8 .b64 bar1;

    mbarrier.init.shared::cta.b64 [bar0], 1;
    mbarrier.init.shared::cta.b64 [bar1], 2;

    mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar0], 128;
    cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [0], [0], 128, [bar0];
    mbarrier.try_wait.parity.shared::cta.b64 %p0, [bar0], 0;

    ret;
}
`
	mod, err := cudaptx.Parse(strings.NewReader(ptx))
	if err != nil {
		t.Fatalf("Parse PTX: %v", err)
	}

	tracker := cudapipeline.NewBarrierTracker()
	violations, err := tracker.Validate(mod)
	if err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if len(violations) > 0 {
		t.Fatalf("unexpected violations: %v", violations)
	}

	bar0 := tracker.Barrier("bar0")
	if bar0 == nil {
		t.Fatal("expected bar0 to be tracked")
	}
	if bar0.Phase != 1 {
		t.Errorf("bar0 phase = %d, want 1", bar0.Phase)
	}
	if bar0.Parity() != 1 {
		t.Errorf("bar0 parity = %d, want 1", bar0.Parity())
	}

	bar1 := tracker.Barrier("bar1")
	if bar1 == nil {
		t.Fatal("expected bar1 to be tracked")
	}
	if bar1.ExpectedArrivals != 2 {
		t.Errorf("bar1 arrivals = %d, want 2", bar1.ExpectedArrivals)
	}
	if bar1.Phase != 0 {
		t.Errorf("bar1 phase = %d, want 0", bar1.Phase)
	}

	all := tracker.Barriers()
	if len(all) != 2 {
		t.Errorf("tracked %d barriers, want 2", len(all))
	}
}

func TestValidatePTX_NilModule(t *testing.T) {
	_, err := cudapipeline.ValidatePTX(nil)
	if err == nil {
		t.Fatal("expected error on nil module, got nil")
	}
}

func TestValidateKernel_NilKernel(t *testing.T) {
	_, err := cudapipeline.ValidateKernel(nil)
	if err == nil {
		t.Fatal("expected error on nil kernel, got nil")
	}
}
