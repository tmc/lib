package cudamega

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// MoEOperation is a detached description of one typed expert operation.
// Gate/up reads the invocation's external hidden vector and slab; Combine
// reads the named expert task outputs. It describes arithmetic, not a launch.
type MoEOperation struct {
	Kind   Opcode
	Config MoEWeightConfig
	Expert int // Gate/up expert index; -1 for Combine.
}

// MoEGateUpOperation seals the real E4M3FN-interleave8 gate/up operation.
func MoEGateUpOperation(config MoEWeightConfig, expert int) (Operation, error) {
	if err := config.validate(); err != nil {
		return Operation{}, err
	}
	if expert < 0 || expert >= config.Experts {
		return Operation{}, fmt.Errorf("expert %d outside 0..%d", expert, config.Experts-1)
	}
	d := MoEOperation{Kind: MoEGateUp, Config: config, Expert: expert}
	return Operation{Kind: MoEGateUp, typedMoE: &d}, nil
}

// MoECombineOperation seals an ordered routing-weighted sum of the one or
// two expert outputs named by the task's Inputs. This is an intermediate
// activation, not a down projection or an entire MoE block.
func MoECombineOperation(config MoEWeightConfig) (Operation, error) {
	if err := config.validate(); err != nil {
		return Operation{}, err
	}
	d := MoEOperation{Kind: MoECombine, Config: config, Expert: -1}
	return Operation{Kind: MoECombine, typedMoE: &d}, nil
}

// MoE returns the sealed typed descriptor, or false for an ordinary or legacy
// device-only operation. The returned value contains no retained pointers.
func (op Operation) MoE() (MoEOperation, bool) {
	if op.typedMoE == nil {
		return MoEOperation{}, false
	}
	return *op.typedMoE, true
}

func (p *Program) hasTypedMoE() bool {
	return len(p.steps) != 0 && p.steps[0].Operation.typedMoE != nil
}

func validateTypedMoE(tasks []cudataskgraph.Task, operations map[string]Operation, grid Grid) error {
	var config MoEWeightConfig
	typed := false
	for _, task := range tasks {
		op := operations[task.ID]
		if op.typedMoE == nil {
			continue
		}
		d := *op.typedMoE
		if err := d.Config.validate(); err != nil {
			return err
		}
		if op.Kind != d.Kind || op.Body != nil || op.Scalar != 0 || op.moe != nil {
			return fmt.Errorf("task %q has an ambiguous typed operation", task.ID)
		}
		if typed && config != d.Config {
			return fmt.Errorf("typed tasks have different MoE configurations")
		}
		config, typed = d.Config, true
		if task.Scope != cudataskgraph.ScopeDevice || task.Extent != uint64(config.Intermediate) || grid.MaxElements != uint32(config.Intermediate) {
			return fmt.Errorf("task %q requires device scope and exact declared intermediate extent", task.ID)
		}
		switch d.Kind {
		case MoEGateUp:
			if d.Expert < 0 || d.Expert >= config.Experts || len(task.Inputs) != 0 {
				return fmt.Errorf("task %q has invalid expert or data inputs", task.ID)
			}
		case MoECombine:
			if d.Expert != -1 || len(task.Inputs) < 1 || len(task.Inputs) > config.Experts {
				return fmt.Errorf("task %q requires one or two expert inputs", task.ID)
			}
			seen := make(map[int]bool)
			for _, name := range task.Inputs {
				input := operations[name].typedMoE
				if input == nil || input.Kind != MoEGateUp || input.Config != config || seen[input.Expert] {
					return fmt.Errorf("task %q input %q is not a distinct compatible expert", task.ID, name)
				}
				seen[input.Expert] = true
			}
		default:
			return fmt.Errorf("task %q has unsupported typed operation", task.ID)
		}
	}
	if typed {
		for _, task := range tasks {
			if operations[task.ID].typedMoE == nil {
				return fmt.Errorf("task %q mixes typed and scalar operations", task.ID)
			}
		}
	}
	return nil
}

// NewMoEMixtureProgram builds two real experts and their routing-weighted
// intermediate activation. The same graph supports one or more worker
// groups, including a one-thread group. Actual CUDA residency is unmeasured.
func NewMoEMixtureProgram(config MoEWeightConfig, grid Grid, target Target) (*Program, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}
	g, err := cudataskgraph.New([]cudataskgraph.Task{
		{ID: "expert-0", Scope: cudataskgraph.ScopeDevice, Extent: uint64(config.Intermediate)},
		{ID: "expert-1", Scope: cudataskgraph.ScopeDevice, Extent: uint64(config.Intermediate)},
		{ID: "mixture", Rank: 1, Scope: cudataskgraph.ScopeDevice, Deps: []string{"expert-0", "expert-1"}, Inputs: []string{"expert-0", "expert-1"}, Extent: uint64(config.Intermediate)},
	})
	if err != nil {
		return nil, err
	}
	if grid.Policy != CountedPolicy {
		return nil, fmt.Errorf("mixture requires counted ownership")
	}
	a, _ := MoEGateUpOperation(config, 0)
	b, _ := MoEGateUpOperation(config, 1)
	c, _ := MoECombineOperation(config)
	return NewProgram(g, map[string]Operation{"expert-0": a, "expert-1": b, "mixture": c}, grid, target)
}
