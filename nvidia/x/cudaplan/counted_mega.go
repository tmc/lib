package cudaplan

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
)

// These wire values match the scalar worker protocol. cudamega imports
// cudamoe, whose stream completions import cudaplan, so importing cudamega
// here would create a cycle. An external test checks the shared wire values.
const (
	countedSpinLimit          = 1 << 24
	countedAbortBadLaunch     = 2
	countedAbortSpinExhausted = 4
	countedAbortBadCompletion = 5
)

// WithCountedMega requests a cooperative FP32 worker with 2..1024 CTAs.
// Each CTA computes a disjoint stripe of every step, publishes its completion,
// and waits for all producers before consuming the next step. The graph subset
// is the same as WithFiniteMega. Graph replay and finite Mega are refused.
// Execution requires cooperative launch support and measured residency of the
// compiled worker; the SM count alone does not admit a launch.
func WithCountedMega(ctas uint32) PlanOption {
	return func(c *planConfig) error {
		if ctas < 2 || ctas > 1024 || c.finiteMega {
			return fmt.Errorf("%w: counted Mega requires 2..1024 CTAs and its own execution mode", ErrUnsupportedBackend)
		}
		c.finiteMega = true
		c.countedCTAs = ctas
		return nil
	}
}

func megaStride(ctas uint32) uint32 {
	if ctas == 0 {
		return 256
	}
	return 256 * ctas
}
func megaIndex(ctas uint32) string {
	if ctas == 0 {
		return "mov.u32 i, %tid.x;\n"
	}
	return "mov.u32 i, %ctaid.x;\nmul.lo.u32 i, i, 256;\nmov.u32 thread, %tid.x;\nadd.u32 i, i, thread;\n"
}

func countedEntry(ctas uint32) string {
	return fmt.Sprintf(`.reg .u64 state_ptr, counter;
.reg .u32 thread, count, spins, old, phase, phase_address;
.shared .align 4 .b8 phase_word[4];
ld.param.u64 state_ptr, [state];
mov.u32 phase_address, phase_word;
mov.u32 thread, %%nctaid.x;
setp.ne.u32 pred, thread, %d;
@pred bra counted_bad_launch;
mov.u32 thread, %%ntid.x;
setp.ne.u32 pred, thread, 256;
@pred bra counted_bad_launch;
mov.u32 thread, %%nctaid.y;
setp.ne.u32 pred, thread, 1;
@pred bra counted_bad_launch;
mov.u32 thread, %%nctaid.z;
setp.ne.u32 pred, thread, 1;
@pred bra counted_bad_launch;
mov.u32 thread, %%ntid.y;
setp.ne.u32 pred, thread, 1;
@pred bra counted_bad_launch;
mov.u32 thread, %%ntid.z;
setp.ne.u32 pred, thread, 1;
@pred bra counted_bad_launch;
`, ctas)
}

// Only thread zero polls. Its verdict reaches the whole CTA through shared
// memory and a barrier, so observing an abort cannot strand sibling threads
// at the next barrier. Every thread releases its output before publication.
func countedBarrier(step int, ctas uint32) string {
	return fmt.Sprintf(`membar.gl;
bar.sync 0;
mov.u32 thread, %%tid.x;
setp.ne.u32 pred, thread, 0;
@pred bra counted_wait_%[1]d;
add.u64 counter, state_ptr, %[2]d;
atom.global.add.u32 old, [counter], 1;
mov.u32 spins, 0;
counted_poll_%[1]d:
ld.acquire.gpu.u32 phase, [state_ptr];
setp.ne.u32 pred, phase, 0;
@pred bra counted_publish_%[1]d;
ld.acquire.gpu.u32 count, [counter];
setp.gt.u32 pred, count, %[3]d;
@pred bra counted_poison_%[1]d;
setp.eq.u32 pred, count, %[3]d;
@pred bra counted_publish_%[1]d;
add.u32 spins, spins, 1;
setp.ge.u32 pred, spins, %[4]d;
@pred bra counted_timeout_%[1]d;
bra counted_poll_%[1]d;
counted_poison_%[1]d:
atom.global.cas.b32 old, [state_ptr], 0, %[5]d;
mov.u32 phase, 1;
bra counted_publish_%[1]d;
counted_timeout_%[1]d:
atom.global.cas.b32 old, [state_ptr], 0, %[6]d;
mov.u32 phase, 1;
counted_publish_%[1]d:
st.shared.u32 [phase_address], phase;
counted_wait_%[1]d:
bar.sync 0;
ld.shared.u32 phase, [phase_address];
setp.ne.u32 pred, phase, 0;
@pred bra counted_return;
membar.gl;
`, step, 4*(step+1), ctas, countedSpinLimit, countedAbortBadCompletion, countedAbortSpinExhausted)
}

func countedBadLaunch() string {
	return fmt.Sprintf(`counted_bad_launch:
mov.u32 thread, %%tid.x;
setp.ne.u32 pred, thread, 0;
@pred bra counted_return;
atom.global.cas.b32 old, [state_ptr], 0, %d;
counted_return:
ret;
`, countedAbortBadLaunch)
}

func countedWorkspace(scratch, names, limit int) (int, int, error) {
	bytes := (names + 1) * 4
	if names < 1 || names > 256 || scratch < 0 || scratch > math.MaxInt-3-bytes {
		return 0, 0, ErrWorkspaceBudgetExceeded
	}
	offset := (scratch + 3) &^ 3
	if offset > limit || bytes > limit-offset {
		return 0, 0, ErrWorkspaceBudgetExceeded
	}
	return offset, offset + bytes, nil
}

func countedResidency(ctas uint32, blocks, sm int32, cooperative int) (uint32, error) {
	if cooperative != 1 || blocks <= 0 || sm <= 0 || uint64(blocks)*uint64(sm) > math.MaxUint32 {
		return 0, fmt.Errorf("%w: counted worker has no measured cooperative residency", ErrUnsupportedBackend)
	}
	resident := uint32(blocks) * uint32(sm)
	if ctas < 2 || ctas > resident {
		return resident, fmt.Errorf("%w: %d CTAs exceed measured residency %d", ErrUnsupportedBackend, ctas, resident)
	}
	return resident, nil
}

func (p *ExecutablePlan) admitCounted(kernel *cuda.Kernel) error {
	return p.dev.WithContext(func() error {
		cooperative, err := p.dev.Attribute(driver.DevAttrCooperativeLaunch)
		if err != nil {
			return err
		}
		sm, err := p.dev.MultiprocessorCount()
		if err != nil {
			return err
		}
		blocks, err := kernel.Raw().MaxActiveBlocksPerMultiprocessor(256, 0)
		if err != nil {
			return err
		}
		if sm > math.MaxInt32 {
			return ErrUnsupportedBackend
		}
		_, err = countedResidency(p.countedCTAs, blocks, int32(sm), cooperative)
		return err
	})
}

func (p *ExecutablePlan) countedOutcome() error {
	bytes := make([]byte, (len(p.megaNames)+1)*4)
	if err := p.workspace.Buffer().Slice(p.megaStateOffset, p.megaStateOffset+len(bytes)).CopyToHost(bytes); err != nil {
		return fmt.Errorf("cudaplan: counted worker outcome: %w", err)
	}
	return p.validateCountedOutcome(bytes)
}

func (p *ExecutablePlan) validateCountedOutcome(bytes []byte) error {
	if len(bytes) != (len(p.megaNames)+1)*4 {
		return fmt.Errorf("cudaplan: truncated counted worker outcome")
	}
	if code := binary.LittleEndian.Uint32(bytes); code != 0 {
		return fmt.Errorf("cudaplan: counted worker aborted with code %d", code)
	}
	for i, n := range p.snap.Order() {
		want := p.countedCTAs
		if n.Kind == cudair.KindInput {
			want = 0
		}
		if got := binary.LittleEndian.Uint32(bytes[4*(i+1):]); got != want {
			return fmt.Errorf("cudaplan: counted worker node %s completed %d CTAs, want %d", n.Name, got, want)
		}
	}
	return nil
}
