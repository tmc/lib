package cudasearch

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudatrial"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type receiptErrorExecutable struct {
	*testExecutable
	raw     cudatrial.Receipt
	failure error
}

func (x *receiptErrorExecutable) Run(context.Context) (cudatrial.Receipt, error) {
	return x.raw, x.failure
}

func TestExecutableReceiptErrorIdentity(t *testing.T) {
	checkOutputLibraries(t)
	for _, test := range []struct {
		name                                         string
		complete, receiptError, goError, copyRefusal bool
	}{
		{name: "receipt-only", receiptError: true},
		{name: "both-errors", complete: true, receiptError: true, goError: true},
		{name: "copy-refusal-receipt", receiptError: true, copyRefusal: true},
		{name: "copy-refusal-both", receiptError: true, goError: true, copyRefusal: true},
		{name: "complete-with-error", complete: true, receiptError: true},
		{name: "incomplete-without-error"},
	} {
		t.Run(test.name, func(t *testing.T) {
			e, baseline, compiled, _ := executableFixture(t)
			s := executableState()
			ctx, b := executableMemoryContext(1 << 20)
			receiptFailure := errors.New("receipt channel failure")
			goFailure := errors.New("Go return channel failure")
			x := &receiptErrorExecutable{testExecutable: compiled, raw: cudatrial.Receipt{Complete: test.complete, Digest: "raw-result", TimingKind: cudatrial.TimingSummedKernel, Timings: map[cudatrial.TimingKind]time.Duration{cudatrial.TimingSummedKernel: 0}, Provenance: map[string]string{"raw": "producer"}, Before: cudatrial.Conditions{Values: map[cudatrial.ConditionKind]int64{cudatrial.ConditionHostAvailableBytes: 0}}}}
			if test.receiptError {
				x.raw.Err = receiptFailure
			}
			if test.goError {
				x.failure = goFailure
			}
			if test.copyRefusal {
				x.raw.Provenance["large"] = strings.Repeat("x", 2<<20)
			}
			e.Compile = func(context.Context, *egraph.Expr, WorkloadBindings) (Executable, error) { return x, nil }
			c, err := e.Evaluate(ctx, s, s.Roots["root"])
			if c != nil || err == nil {
				if c != nil {
					c.Close()
				}
				t.Fatal("failed receipt admitted", c, err)
			}
			for _, channel := range []struct {
				present bool
				failure error
			}{{test.receiptError, receiptFailure}, {test.goError, goFailure}} {
				if errors.Is(err, channel.failure) != channel.present {
					t.Fatalf("returned error lost or invented producer identity: %v", err)
				}
			}
			if test.copyRefusal {
				if !errors.Is(err, ErrMemoryBudgetExceeded) || len(e.Receipts) != 1 || !e.Receipts[0].Complete {
					t.Fatal("copy refusal lost admitted prefix or error", err, e.Receipts)
				}
				// Refused metadata is not fabricated as a detached trial or mutated in its
				// borrowed producer owner. Both error identities must still be returned.
				if x.raw.Digest != "raw-result" || x.raw.Provenance["raw"] != "producer" || len(x.raw.Provenance["large"]) != 2<<20 || x.raw.Complete != test.complete || errors.Is(x.raw.Err, receiptFailure) != test.receiptError {
					t.Fatal("copy refusal mutated raw producer receipt")
				}
			} else {
				if len(e.Receipts) != 2 {
					t.Fatal("failed trial missing", e.Receipts)
				}
				raw := e.Receipts[1]
				if raw.Complete || raw.Digest != "raw-result" || raw.Provenance["raw"] != "producer" || !raw.Before.Has(cudatrial.ConditionHostAvailableBytes) || raw.Before.Values[cudatrial.ConditionHostAvailableBytes] != 0 || !raw.HasTiming(cudatrial.TimingSummedKernel) || raw.Timings[cudatrial.TimingSummedKernel] != 0 {
					t.Fatalf("failed receipt fields or measured zero changed: %+v", raw)
				}
				for _, channel := range []struct {
					present bool
					failure error
				}{{test.receiptError, receiptFailure}, {test.goError, goFailure}} {
					if errors.Is(raw.Err, channel.failure) != channel.present {
						t.Fatal("retained trial error lost producer identity", raw.Err)
					}
				}
				if !test.receiptError && !test.goError && (!strings.Contains(err.Error(), "incomplete execution receipt") || !strings.Contains(raw.Err.Error(), "incomplete execution receipt")) {
					t.Fatal("incomplete receipt lost generic refusal", err, raw.Err)
				}
				x.raw.Provenance["raw"] = "mutated"
				x.raw.Before.Values[cudatrial.ConditionHostAvailableBytes] = 99
				if raw.Provenance["raw"] != "producer" || raw.Before.Values[cudatrial.ConditionHostAvailableBytes] != 0 {
					t.Fatal("error receipt lost detachment")
				}
			}
			if compiled.closed != 1 || baseline.closed != 0 {
				t.Fatal("failed attempt ownership changed", compiled.closed, baseline.closed)
			}
			if err := e.Close(); err != nil || b.live != 0 {
				t.Fatal("failed receipt metadata leaked", err, b.live)
			}
			if err := e.Close(); err != nil || b.live != 0 {
				t.Fatal("close not idempotent", err, b.live)
			}
		})
	}
}
