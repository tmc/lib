package cudareg

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudadevicetable"
	"github.com/tmc/lib/nvidia/x/cudagroup"
)

// targetPattern matches an exact PTX target such as "sm_90a".
var targetPattern = regexp.MustCompile(`^sm_[0-9]+[a-z]*$`)

// Action is a register ownership change.
type Action uint8

const (
	Decrease Action = iota + 1
	Increase
)

func (a Action) String() string {
	switch a {
	case Decrease:
		return "dec"
	case Increase:
		return "inc"
	default:
		return "unknown"
	}
}

// Target describes the exact PTX target suffix admitted by this package.
type Target struct {
	Name        string
	LegalCounts []uint32
}

// The setmaxnreg operand range and granularity. These were measured rather
// than copied from a document: on ptxas 13.0.88 both .dec and .inc accept 24
// and 256 and reject 23, 25, 28 and 264, identically at sm_90a and sm_121a.
// TestLegalCountBoundaries re-runs that check wherever a ptxas is available.
const (
	minRegisterCount  = 24
	maxRegisterCount  = 256
	registerCountStep = 8
)

// TargetFor returns the setmaxnreg capability for an exact target. Family
// names are rejected, and so are unsuffixed targets that the table records a
// rejection for: sm_90 is real silicon that ptxas refuses setmaxnreg on.
//
// Which targets those are is read from the probed feature table rather than
// listed here. The list this replaced named sm_90a, sm_100a, sm_110a and
// sm_120a and omitted sm_121a, so the package that models setmaxnreg refused
// the only architecture this project has ever attached -- while ptxas accepts
// setmaxnreg there.
func TargetFor(name string) (Target, error) {
	if !targetPattern.MatchString(strings.ToLower(strings.TrimSpace(name))) {
		return Target{}, fmt.Errorf("unsupported setmaxnreg target %q: want an exact target such as \"sm_90a\"", name)
	}
	// A target the table does not cover is admitted, not refused. setmaxnreg
	// is an instruction: if a future part lacks it, ptxas says so at assembly
	// time with a better message than this package could invent. Refusing on
	// silence had the opposite failure -- it would make each new GPU less
	// capable than its predecessor until someone regenerated a table, and the
	// unsuffixed rejection below is the case that actually needs enforcing.
	// See [cudadevicetable.AssumeSupported].
	a := cudadevicetable.Ask(name, "setmaxnreg")
	if !a.Or(cudadevicetable.AssumeSupported) {
		return Target{}, fmt.Errorf("unsupported setmaxnreg target %q: %s", name, a.Why())
	}
	counts := make([]uint32, 0, 30)
	for n := uint32(minRegisterCount); n <= maxRegisterCount; n += registerCountStep {
		counts = append(counts, n)
	}
	return Target{Name: name, LegalCounts: counts}, nil
}

// Budget separates compiler allocation facts from dynamic ownership.
type Budget struct {
	Group             cudagroup.Group
	CompilerAllocated uint32 // registers allocated per thread by compiler/driver
	PeakLive          uint32 // analysis peak, not an allocation request
	Cap               uint32 // compiler or launch cap for this role
	Dynamic           uint32 // current setmaxnreg ownership per thread
}

// Transition is one uniform warpgroup setmaxnreg operation.
type Transition struct {
	Role         string
	Action       Action
	Count        uint32
	Participants uint32
	Synchronized bool
	Initialized  bool // required after an increase before new registers are used
}

// Schedule is a finite CTA-local register redistribution protocol.
type Schedule struct {
	Target          Target
	CTARegisterPool uint64 // total register units available to this CTA
	Groups          map[string]Budget
	Steps           []Transition
}

// Validate checks target, resource facts, exact warpgroup uniformity, pool
// conservation, transition direction, and the explicit progress path.
func (s Schedule) Validate() error {
	if s.Target.Name == "" {
		return errors.New("register schedule has no target")
	}
	cap, err := TargetFor(s.Target.Name)
	if err != nil {
		return err
	}
	if len(s.Groups) == 0 {
		return errors.New("register schedule has no groups")
	}
	roles := make([]string, 0, len(s.Groups))
	if len(s.Groups) > 8 {
		return errors.New("register schedule exceeds eight warpgroups")
	}
	for role := range s.Groups {
		roles = append(roles, role)
	}
	for i := range roles {
		for j := i + 1; j < len(roles); j++ {
			a, b := s.Groups[roles[i]].Group, s.Groups[roles[j]].Group
			ar, err := a.Rank(cudagroup.Dim3{})
			if err != nil {
				return fmt.Errorf("group %q rank: %w", roles[i], err)
			}
			br, err := b.Rank(cudagroup.Dim3{})
			if err != nil {
				return fmt.Errorf("group %q rank: %w", roles[j], err)
			}
			if ar < br+128 && br < ar+128 {
				return fmt.Errorf("groups %q and %q overlap physically", roles[i], roles[j])
			}
		}
	}
	if s.CTARegisterPool == 0 {
		return errors.New("register pool is zero")
	}
	current := make(map[string]uint32, len(s.Groups))
	var owned uint64
	for role, b := range s.Groups {
		if role == "" {
			return errors.New("register group role is empty")
		}
		if b.CompilerAllocated == 0 || b.Cap == 0 || b.Dynamic == 0 {
			return fmt.Errorf("group %q has zero register fact", role)
		}
		if b.Group.Kind() != cudagroup.WarpGroup || b.Group.Size() != 128 {
			return fmt.Errorf("group %q is not a complete warpgroup", role)
		}
		if b.PeakLive > b.Cap {
			return fmt.Errorf("group %q peak live registers exceed cap", role)
		}
		if b.Dynamic > b.Cap {
			return fmt.Errorf("group %q dynamic ownership exceeds cap", role)
		}
		if b.CompilerAllocated != b.Dynamic {
			return fmt.Errorf("group %q compiler allocation %d does not match initial dynamic ownership %d", role, b.CompilerAllocated, b.Dynamic)
		}
		if !legal(cap, b.Dynamic) {
			return fmt.Errorf("group %q has illegal register count %d", role, b.Dynamic)
		}
		current[role] = b.Dynamic
		owned += uint64(b.Dynamic) * 128
	}
	if owned != s.CTARegisterPool {
		return fmt.Errorf("CTA register pool %d does not equal initial ownership %d", s.CTARegisterPool, owned)
	}
	pool := s.CTARegisterPool - owned
	for i, step := range s.Steps {
		b, ok := s.Groups[step.Role]
		if !ok {
			return fmt.Errorf("step %d names unknown group %q", i, step.Role)
		}
		if step.Participants != 128 {
			return fmt.Errorf("step %d requires all 128 warpgroup participants", i)
		}
		if !step.Synchronized {
			return fmt.Errorf("step %d lacks explicit warpgroup synchronization", i)
		}
		if step.Action != Decrease && step.Action != Increase {
			return fmt.Errorf("step %d has invalid direction", i)
		}
		if !legal(cap, step.Count) {
			return fmt.Errorf("step %d has illegal register count %d", i, step.Count)
		}
		if step.Count > b.Cap {
			return fmt.Errorf("step %d exceeds group %q cap", i, step.Role)
		}
		before := current[step.Role]
		switch step.Action {
		case Decrease:
			if step.Count > before {
				return fmt.Errorf("step %d decreases from %d to %d", i, before, step.Count)
			}
			pool += uint64(before-step.Count) * 128
		case Increase:
			if step.Count < before {
				return fmt.Errorf("step %d increases from %d to %d", i, before, step.Count)
			}
			delta := uint64(step.Count-before) * 128
			if delta > pool {
				return fmt.Errorf("step %d blocks without a donor release", i)
			}
			pool -= delta
			if !step.Initialized {
				return fmt.Errorf("step %d uses uninitialized registers after increase", i)
			}
		}
		current[step.Role] = step.Count
		_ = b
	}
	return nil
}

func legal(t Target, n uint32) bool {
	for _, x := range t.LegalCounts {
		if x == n {
			return true
		}
	}
	return false
}

// GenerateFixture returns a finite producer release and consumer growth
// schedule. Its generated source uses only the documented PTX instruction.
func GenerateFixture() (Schedule, string, error) {
	t, err := TargetFor("sm_90a")
	if err != nil {
		return Schedule{}, "", err
	}
	cta, _ := cudagroup.New(cudagroup.CTA, cudagroup.Dim3{X: 256, Y: 1, Z: 1})
	producer, _ := cudagroup.WarpGroupForCTA(cta, 0)
	consumer, _ := cudagroup.WarpGroupForCTA(cta, 4)
	s := Schedule{Target: t, CTARegisterPool: 16384, Groups: map[string]Budget{
		"producer": {Group: producer, CompilerAllocated: 64, PeakLive: 32, Cap: 64, Dynamic: 64},
		"consumer": {Group: consumer, CompilerAllocated: 64, PeakLive: 64, Cap: 96, Dynamic: 64},
	}, Steps: []Transition{{Role: "producer", Action: Decrease, Count: 32, Participants: 128, Synchronized: true}, {Role: "consumer", Action: Increase, Count: 96, Participants: 128, Synchronized: true, Initialized: true}}}
	if err := s.Validate(); err != nil {
		return Schedule{}, "", err
	}
	src := "// target=sm_90a; generated finite producer-consumer fixture; compile with -maxrregcount=64\nextern \"C\" __global__ void cudareg_fixture(const unsigned *in, unsigned *out) {\n  unsigned tid = threadIdx.x;\n  if (tid < 128) { unsigned x = in[tid]; asm(\"setmaxnreg.dec.sync.aligned.u32 32;\"); out[tid] = x + 1; }\n  __syncthreads();\n  if (tid >= 128) { asm(\"setmaxnreg.inc.sync.aligned.u32 96;\"); unsigned x = in[tid]; out[tid] = x + 1; }\n  __syncthreads();\n}\n"
	return s, strings.TrimSpace(src), nil
}

func warpgroup() cudagroup.Group {
	g, _ := cudagroup.New(cudagroup.WarpGroup, cudagroup.Dim3{X: 128, Y: 1, Z: 1})
	return g
}

// Reference applies the fixture's producer-consumer operation on the host.
// It is an independent, fixed-budget baseline for output comparisons.
func Reference(input []uint32) []uint32 {
	out := make([]uint32, len(input))
	for i, v := range input {
		out[i] = v + 1
	}
	return out
}

// FixedBudgetReference is the ordinary register-budget baseline. It has the
// same computation and therefore isolates register protocol effects.
func FixedBudgetReference(input []uint32) []uint32 { return Reference(input) }

// FixedBudgetSource returns the ordinary baseline kernel used for output
// comparison. It performs the same operation without dynamic redistribution.
func FixedBudgetSource() string {
	return `extern "C" __global__ void cudareg_fixed(const unsigned *in, unsigned *out) {
  out[threadIdx.x] = in[threadIdx.x] + 1;
}`
}
