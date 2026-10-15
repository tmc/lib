package cudapipeline_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudapipeline"
)

const transactionInit = "mbarrier.init.shared::cta.b64 [bar], 1;\n"

func transactionTensor(name string) string {
	return fmt.Sprintf("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [0], [%s, {0, 0}], [bar];\n", name)
}

func transactionExpect(bytes string) string {
	return "mbarrier.arrive.expect_tx.shared::cta.b64 %state, [bar], " + bytes + ";\n"
}

func transactionComplete(bytes string) string {
	return "mbarrier.complete_tx.shared::cta.b64 [bar], " + bytes + ";\n"
}

func transactionBulk(bytes string) string {
	return "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [0], [0], " + bytes + ", [bar];\n"
}

func transactionWaits() []string {
	return []string{
		"mbarrier.try_wait.parity.shared::cta.b64 %p, [bar], 0;",
		"mbarrier.test_wait.parity.shared::cta.b64 %p, [bar], 0;",
		"mbarrier.try_wait.shared::cta.b64 %p, [bar], %state;",
		"mbarrier.test_wait.shared::cta.b64 %p, [bar], %state;",
	}
}

func TestTransactionUnknownSizes(t *testing.T) {
	for _, wait := range transactionWaits() {
		for _, tc := range []struct{ name, body string }{
			{"tensor", transactionExpect("16") + transactionTensor("unknown")},
			{"mixed unequal subtotal", transactionExpect("32") + transactionTensor("known") + transactionTensor("unknown")},
			{"mixed equal subtotal", transactionExpect("16") + transactionTensor("unknown") + transactionTensor("known")},
			{"expect", transactionExpect("%bytes") + transactionTensor("known")},
			{"standalone expect", "mbarrier.expect_tx.shared::cta.b64 [bar], %bytes;\nmbarrier.arrive.shared::cta.b64 %state, [bar];\n" + transactionTensor("known")},
			{"complete", transactionExpect("16") + transactionComplete("%bytes")},
			{"bulk", transactionExpect("16") + transactionBulk("%bytes")},
		} {
			t.Run(wait+"/"+tc.name, func(t *testing.T) {
				k := parseControlKernel(t, transactionInit+tc.body+wait+"\nret;")
				violations, err := cudapipeline.ValidateKernel(k, cudapipeline.WithTensorSize("known", 16))
				if err != nil || len(violations) != 1 {
					t.Fatalf("violations %v, error %v", violations, err)
				}
				v := violations[0]
				want := k.Instructions[len(k.Instructions)-2]
				if v.Violation != cudapipeline.ErrUnverifiedTxBytes || v.Line != want.Line || v.Opcode != want.Opcode || v.Barrier != "bar" || v.Phase != 0 || !strings.Contains(v.Message, "known subtotals") {
					t.Fatalf("missing located unverified wait diagnostic: %+v", v)
				}
			})
		}
	}
}

func TestTransactionKnownSizes(t *testing.T) {
	for _, wait := range transactionWaits() {
		for _, tc := range []struct {
			name, body       string
			mismatch         bool
			expected, actual int64
		}{
			{"tensor", transactionExpect("16") + transactionTensor("known"), false, 16, 16},
			{"known zero", transactionExpect("0") + transactionTensor("zero"), false, 0, 0},
			{"zero completion", transactionExpect("16") + transactionTensor("zero"), true, 16, 0},
			{"undercount", transactionExpect("32") + transactionTensor("known"), true, 32, 16},
			{"overcount", transactionExpect("8") + transactionTensor("known"), true, 8, 16},
			{"no copy", transactionExpect("16"), true, 16, 0},
			{"complete", transactionExpect("16") + transactionComplete("16"), false, 16, 16},
			{"bulk", transactionExpect("16") + transactionBulk("16"), false, 16, 16},
			{"summed expectations", "mbarrier.expect_tx.shared::cta.b64 [bar], 16;\n" + transactionExpect("16") + transactionTensor("known") + transactionBulk("16"), false, 32, 32},
			{"int64 boundary", transactionExpect(fmt.Sprint(int64(math.MaxInt64))) + transactionComplete(fmt.Sprint(int64(math.MaxInt64))), false, math.MaxInt64, math.MaxInt64},
		} {
			t.Run(wait+"/"+tc.name, func(t *testing.T) {
				violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, transactionInit+tc.body+wait+"\nret;"),
					cudapipeline.WithTensorSize("known", 16), cudapipeline.WithTensorSize("zero", 0))
				if err != nil {
					t.Fatal(err)
				}
				if !tc.mismatch {
					if len(violations) != 0 {
						t.Fatal(violations)
					}
					return
				}
				if len(violations) != 1 || violations[0].Violation != cudapipeline.ErrMismatchedTxBytes || violations[0].Expected != tc.expected || violations[0].Actual != tc.actual {
					t.Fatalf("known mismatch: %v", violations)
				}
			})
		}
	}
}

func TestTransactionSizeStateAndReset(t *testing.T) {
	for _, wait := range transactionWaits() {
		t.Run(wait, func(t *testing.T) { testTransactionSizeStateAndReset(t, wait) })
	}
}

func testTransactionSizeStateAndReset(t *testing.T, wait string) {
	t.Helper()
	tracker := cudapipeline.NewBarrierTracker(cudapipeline.WithTensorSize("known", 16))
	body := transactionInit + transactionExpect("16") + "mbarrier.expect_tx.shared::cta.b64 [bar], %missing;\n" + transactionTensor("unknown") + transactionTensor("known")
	violations, err := tracker.ValidateKernel(parseControlKernel(t, body+"ret;"))
	if err != nil || len(violations) != 0 {
		t.Fatalf("pre-wait state: %v, %v", violations, err)
	}
	b := tracker.Barrier("bar")
	if b == nil || b.ExpectedTxBytes != 16 || b.CompletedTxBytes != 16 || !b.ExpectedTxBytesUnknown || !b.CompletedTxBytesUnknown || b.CopyCount != 2 {
		t.Fatalf("unknown size was lost or assigned a fabricated extent: %+v", b)
	}
	var unknown, known, unknownExpected int
	for _, event := range b.Events {
		if event.Kind == cudapipeline.EventExpectTx && event.BytesUnknown {
			unknownExpected++
		}
		if event.Kind == cudapipeline.EventCopy {
			if event.BytesUnknown {
				unknown++
				if event.Bytes != 0 {
					t.Fatal("unknown copy acquired invented bytes")
				}
			} else if event.Bytes == 16 {
				known++
			}
		}
	}
	if unknown != 1 || known != 1 || unknownExpected != 1 {
		t.Fatalf("copy event metadata: %+v", b.Events)
	}
	// The first wait cannot certify its equal known subtotals. Its uncertainty
	// must not contaminate the next phase, whose complete extent is known.
	secondWait := strings.Replace(wait, ", 0;", ", 1;", 1)
	body += wait + "\n" + transactionExpect("16") + transactionTensor("known") + secondWait + "\nret;"
	violations, err = tracker.ValidateKernel(parseControlKernel(t, body))
	if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrUnverifiedTxBytes || violations[0].Phase != 0 {
		t.Fatalf("phase-local uncertainty: %v, %v", violations, err)
	}
	b = tracker.Barrier("bar")
	if b.Phase != 2 || b.ExpectedTxBytesUnknown || b.CompletedTxBytesUnknown || b.ExpectedTxBytes != 0 || b.CompletedTxBytes != 0 || b.CopyCount != 0 || b.HasExpectedTx {
		t.Fatalf("phase state was not reset: %+v", b)
	}
	// ValidateKernel resets transient state but keeps the configured size.
	violations, err = tracker.ValidateKernel(parseControlKernel(t, transactionInit+transactionExpect("16")+transactionTensor("known")+wait+"\nret;"))
	if err != nil || len(violations) != 0 {
		t.Fatalf("kernel reset lost known metadata: %v, %v", violations, err)
	}
}

func TestTransactionSizeArithmetic(t *testing.T) {
	max := fmt.Sprint(int64(math.MaxInt64))
	for _, wait := range transactionWaits() {
		for _, tc := range []struct {
			name, body string
			invalid    int
		}{
			{"negative expect", transactionExpect("-1") + transactionComplete("16"), 1},
			{"negative standalone expect", "mbarrier.expect_tx.shared::cta.b64 [bar], -1;\nmbarrier.arrive.shared::cta.b64 %state, [bar];\n", 1},
			{"negative complete", transactionExpect("16") + transactionComplete("-1"), 1},
			{"negative bulk", transactionExpect("16") + transactionBulk("-1"), 1},
			{"negative tensor metadata", transactionExpect("16") + transactionTensor("negative"), 1},
			{"expected overflow", "mbarrier.expect_tx.shared::cta.b64 [bar], " + max + ";\n" + transactionExpect("1"), 1},
			{"completed overflow", transactionExpect(max) + transactionComplete(max) + transactionComplete("1"), 1},
			{"bulk overflow", transactionExpect(max) + transactionBulk(max) + transactionBulk("1"), 1},
			{"tensor overflow", transactionExpect(max) + transactionTensor("max") + transactionTensor("one"), 1},
			{"matching overflow", "mbarrier.expect_tx.shared::cta.b64 [bar], " + max + ";\n" + transactionExpect("1") + transactionComplete(max) + transactionComplete("1"), 2},
		} {
			t.Run(wait+"/"+tc.name, func(t *testing.T) {
				tracker := cudapipeline.NewBarrierTracker(cudapipeline.WithTensorSize("negative", -1), cudapipeline.WithTensorSize("max", math.MaxInt64), cudapipeline.WithTensorSize("one", 1))
				violations, err := tracker.ValidateKernel(parseControlKernel(t, transactionInit+tc.body+wait+"\nret;"))
				if err != nil || len(violations) != tc.invalid+1 {
					t.Fatalf("invalid byte accounting: %v, %v", violations, err)
				}
				invalid, unverified := 0, 0
				for _, v := range violations {
					switch v.Violation {
					case cudapipeline.ErrInvalidTxBytes:
						invalid++
					case cudapipeline.ErrUnverifiedTxBytes:
						unverified++
					default:
						t.Fatalf("partial/invalid totals produced a false exact diagnosis: %+v", v)
					}
				}
				if invalid != tc.invalid || unverified != 1 {
					t.Fatal(violations)
				}
			})
		}
	}
}

func TestTransactionRegisterSizeOverwrite(t *testing.T) {
	for _, overwrite := range []string{
		"mov.u32 %bytes, %dynamic;\n", "ld.param.u32 %bytes, [dynamic];\n",
		"add.u32 %bytes, %dynamic, 1;\n", "ld.global.u32 %bytes, [dynamic];\n",
		"ld.global.v2.u32 {%bytes, %other}, [dynamic];\n",
		"@%p mov.u32 %bytes, 32;\n", "@%p ld.param.u32 %bytes, [dynamic];\n",
	} {
		for _, operation := range []string{
			transactionExpect("%bytes") + transactionComplete("16"),
			transactionExpect("16") + transactionComplete("%bytes"),
			transactionExpect("16") + transactionBulk("%bytes"),
		} {
			body := transactionInit + "mov.u32 %bytes, 16;\n" + overwrite + operation + transactionWaits()[0] + "\nret;"
			violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body))
			if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrUnverifiedTxBytes {
				t.Fatalf("stale numeric fact after %q: %v, %v", overwrite, violations, err)
			}
		}
	}
	body := transactionInit + "mov.u64 %map, known;\nmov.u64 %map, 16;\n" + transactionExpect("16") + transactionTensor("%map") + transactionWaits()[0] + "\nret;"
	violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body), cudapipeline.WithTensorSize("known", 16))
	if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrUnverifiedTxBytes {
		t.Fatalf("stale tensor-map symbol fact: %v, %v", violations, err)
	}
	for _, load := range []string{"@%p mov.u64 %map, known;\n", "@%p ld.param.u64 %map, [known];\n"} {
		body := transactionInit + load + transactionExpect("16") + transactionTensor("%map") + transactionWaits()[0] + "\nret;"
		violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body), cudapipeline.WithTensorSize("known", 16))
		if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrUnverifiedTxBytes {
			t.Fatalf("conditional tensor-map fact from %q: %v, %v", load, violations, err)
		}
	}
}

func TestTransactionCompoundRegisterOverwrite(t *testing.T) {
	for _, overwrite := range []string{
		"ld.param.v2.u32 {%bytes, %other}, [dynamic];\n",
		"mov.b64 {%bytes, %other}, %dynamic64;\n",
		"@%p ld.param.v2.u32 {%bytes, %other}, [dynamic];\n",
		"@%p mov.b64 {%bytes, %other}, %dynamic64;\n",
		// Even a literal compound source does not establish either scalar value.
		"mov.b64 {%bytes, %other}, 16;\n",
	} {
		for _, register := range []string{"%bytes", "%other"} {
			for _, operation := range []string{
				transactionExpect(register) + transactionComplete("16"),
				transactionExpect("16") + transactionComplete(register),
				transactionExpect("16") + transactionBulk(register),
			} {
				for _, wait := range transactionWaits() {
					body := transactionInit + "mov.u32 %bytes, 16;\nmov.u32 %other, 16;\n" + overwrite + operation + wait + "\nret;"
					violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body))
					if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrUnverifiedTxBytes {
						t.Fatalf("compound overwrite %q retained %s for %q and %q: %v, %v", overwrite, register, operation, wait, violations, err)
					}
				}
			}
		}
	}
	for _, overwrite := range []string{
		"ld.param.v2.u64 {%map, %other}, [known];\n",
		"ld.param.v2.u64 {%other, %map}, [known];\n",
		"@%p ld.param.v2.u64 {%map, %other}, [known];\n",
		"@%p ld.param.v2.u64 {%other, %map}, [known];\n",
	} {
		for _, wait := range transactionWaits() {
			body := transactionInit + "mov.u64 %map, known;\n" + overwrite + transactionExpect("16") + transactionTensor("%map") + wait + "\nret;"
			violations, err := cudapipeline.ValidateKernel(parseControlKernel(t, body), cudapipeline.WithTensorSize("known", 16))
			if err != nil || len(violations) != 1 || violations[0].Violation != cudapipeline.ErrUnverifiedTxBytes {
				t.Fatalf("compound overwrite %q retained tensor alias for %q: %v, %v", overwrite, wait, violations, err)
			}
		}
	}
}
