package cudapipeline_test

import (
	"fmt"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
)

func TestPipelineUnknownProtocolOpcode(t *testing.T) {
	for _, statement := range []string{
		"mbarrier.future.shared::cta.b64 [bar];",
		"mbarrier.init.future.shared::cta.b64 [bar], 1;",
		"mbarrier.try_wait.parity.shared::cta.b64.future %p, [bar], 0;",
		"cp.async.bulk.tensor.2d.shared::cta.global.bulk_group [0], [tmap, {0, 0}], [bar];",
		"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.future [0], [0], 16, [bar];",
		"cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.multicast::cluster [0], [tmap, {0, 0}], [bar], 3;",
		"@%p mbarrier.init.shared::cta.b64 [bar], 1;",
		"@%p cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [0], [0], 16, [bar];",
	} {
		t.Run(statement, func(t *testing.T) {
			k := parseControlKernel(t, statement+"\nret;")
			violations, err := cudapipeline.ValidateKernel(k)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, v := range violations {
				if v.Violation == "unsupported pipeline instruction" && v.Opcode == k.Instructions[0].Opcode && v.Line == k.Instructions[0].Line {
					found = true
				}
			}
			if !found {
				t.Fatalf("unmodeled protocol instruction was not reported: %v", violations)
			}
		})
	}
}

func TestPipelineModeledProtocolOpcodes(t *testing.T) {
	const init = "mbarrier.init.shared::cta.b64 [bar], 1;\n"
	for _, wait := range []string{
		"mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;",
		"mbarrier.test_wait.parity.shared::cta.b64 %p, [bar], 0;",
		"mbarrier.try_wait.shared::cta.b64 %p, [bar], %state;",
		"mbarrier.test_wait.shared::cta.b64 %p, [bar], %state;",
	} {
		t.Run(wait, func(t *testing.T) {
			body := init + "mbarrier.arrive.shared::cta.b64 %state, [bar];\n" + wait + "\nmbarrier.inval.shared::cta.b64 [bar];\nret;"
			violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body))
			if err != nil || len(violations) != 0 {
				t.Fatalf("modeled barrier sequence: %v, %v", violations, err)
			}
		})
	}
	for _, copy := range []string{
		cudapipeline.EmitTMABulkAsync2D("0", "tmap", "0", "0", "bar"),
		cudapipeline.EmitTMABulkAsync2DCluster("0", "tmap", "0", "0", "bar"),
		"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [0], [0], 16, [bar];",
	} {
		t.Run(copy, func(t *testing.T) {
			body := init + cudapipeline.EmitArriveExpectTx("bar", 16) + "\n" + copy + "\n" + cudapipeline.EmitWaitParity("bar", 0, "%p") + "\nret;"
			violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body), cudapipeline.WithTensorSize("tmap", 16))
			if err != nil || len(violations) != 0 {
				t.Fatalf("modeled emitted sequence: %v, %v", violations, err)
			}
		})
	}
	// These exact legacy strings remain explicitly opt-in analysis fixtures.
	// No hardware instruction support follows from preserving their model.
	for i, copy := range []string{
		cudapipeline.EmitTMABulkAsync2DTMEM("0", "tmap", "0", "0", "bar"),
		cudapipeline.EmitTMABulkAsyncTMEM("0", "0", 16, "bar"),
	} {
		t.Run(fmt.Sprint("legacy", i), func(t *testing.T) {
			body := init + cudapipeline.EmitArriveExpectTx("bar", 16) + "\n" + copy + "\n" + cudapipeline.EmitWaitParity("bar", 0, "%p") + "\nret;"
			k := parseControlKernel(t, body)
			violations, err := cudapipeline.ValidateKernel(k, cudapipeline.WithTensorSize("tmap", 16), cudapipeline.WithAllowLegacyTMEM(true))
			if err != nil || len(violations) != 0 {
				t.Fatalf("legacy opt-in fixture changed: %v, %v", violations, err)
			}
			violations, err = cudapipeline.ValidateKernel(k, cudapipeline.WithTensorSize("tmap", 16))
			if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrInvalidBulkTMEMPTX {
				t.Fatalf("default legacy refusal changed: %v, %v", violations, err)
			}
		})
	}
	// Standalone expect/complete accounting is already modeled separately.
	body := init + "mbarrier.expect_tx.shared::cta.b64 [bar], 16;\nmbarrier.complete_tx.shared::cta.b64 [bar], 16;\nret;"
	tracker := cudapipeline.NewBarrierTracker()
	violations, err := tracker.ValidateKernel(parseControlKernel(t, body))
	if err != nil || len(violations) != 0 {
		t.Fatalf("standalone transaction accounting changed: %v, %v", violations, err)
	}
	barrier := tracker.Barrier("bar")
	if barrier == nil || barrier.ExpectedTxBytes != 16 || barrier.CompletedTxBytes != 16 {
		t.Fatalf("standalone transaction bytes lost: %+v", barrier)
	}
}

func TestPipelineUnrelatedArithmetic(t *testing.T) {
	body := "mbarrier.init.shared::cta.b64 [bar], 1;\nadd.u32 %r, %r, 1;\nfma.rn.f32 %f, %f, %f, %f;\nunmodeled_arithmetic %r;\nret;"
	violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body))
	if err != nil || len(violations) != 0 {
		t.Fatalf("unrelated arithmetic treated as a protocol instruction: %v, %v", violations, err)
	}
}
