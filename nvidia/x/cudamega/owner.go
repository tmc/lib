package cudamega

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudataskgraph"
)

// An Owner is the half-open range of CTAs that performs a task.
//
// It is a range rather than a set because the emitted worker decides
// membership from blockIdx.x alone, with two comparisons and no table, and
// because a range is what lets the element stride below be a constant: a
// task's elements are dealt to its owners in exactly the way the whole grid
// deals a wave's, with the range standing in for the grid.
//
// The zero Owner owns nothing and is not a value [NewProgram] produces.
type Owner struct {
	Lo, Hi uint32
}

// Count is how many CTAs perform the task.
//
// It is the number a consumer's wait must reach, and it is the reason the
// wait cannot be written against the grid: a task whose owner is a proper
// subrange publishes Count completions and never grid_ctas of them, so a wait
// spelled against the grid is not a slow wait but an unsatisfiable one.
func (o Owner) Count() uint32 { return o.Hi - o.Lo }

// Contains reports whether cta performs the task.
func (o Owner) Contains(cta uint32) bool { return cta >= o.Lo && cta < o.Hi }

func (o Owner) String() string { return fmt.Sprintf("[%d,%d)", o.Lo, o.Hi) }

// A Policy is how tasks are dealt to CTAs.
//
// Both policies read the graph's shape -- how many tasks share a rank, and
// how many CTAs the grid has -- and nothing else. In particular neither reads
// [cudataskgraph.Task.Bytes]: that field is a dynamic shared-memory claim,
// not an output size and not a cost, and a policy that ranked tasks by it
// would be ranking them by a quantity that says nothing about them. See
// [Policy.Assign] and TestSelectionIgnoresBytes.
type Policy int

const (
	// WavePolicy gives every CTA a share of every task's elements. It is the
	// zero value, and it is what this package emitted before per-task
	// ownership existed: one wave per task, the whole grid in each.
	WavePolicy Policy = iota

	// CountedPolicy partitions the grid among the tasks of each rank, so that
	// two independent tasks of one rank run at the same time on disjoint
	// CTAs. It is the policy the dependency wait is actually for: under it a
	// consumer generally reads elements a different CTA wrote.
	CountedPolicy
)

func (p Policy) String() string {
	switch p {
	case WavePolicy:
		return "wave"
	case CountedPolicy:
		return "counted"
	}
	return fmt.Sprintf("Policy(%d)", int(p))
}

// Assign deals tasks to CTAs. Tasks must be in the order [cudataskgraph.Graph.Tasks]
// returns them, which is rank order.
//
// Under [CountedPolicy] the CTAs of a rank of width w are split into w
// contiguous ranges by j*ctas/w, which is the partition that stays a
// partition for every w and every ctas: the ranges abut, the first starts at
// zero and the last ends at ctas, so every element of every task has exactly
// one owner. A rank wider than the grid is dealt as a wave instead -- the
// partition would leave some task of that rank with an empty range, and a
// task no CTA performs is a task whose completion word never moves.
//
// Nothing here reads Task.Bytes, Task.Scope or Task.Deps. Rank and count are
// the whole input.
func (p Policy) Assign(tasks []cudataskgraph.Task, ctas uint32) ([]Owner, error) {
	if ctas == 0 {
		return nil, fmt.Errorf("cannot deal %d tasks to an empty grid", len(tasks))
	}
	owners := make([]Owner, len(tasks))
	switch p {
	case WavePolicy:
		for i := range owners {
			owners[i] = Owner{Lo: 0, Hi: ctas}
		}
	case CountedPolicy:
		for lo := 0; lo < len(tasks); {
			hi := lo
			for hi < len(tasks) && tasks[hi].Rank == tasks[lo].Rank {
				hi++
			}
			w := uint32(hi - lo)
			for j := lo; j < hi; j++ {
				if w > ctas {
					owners[j] = Owner{Lo: 0, Hi: ctas}
					continue
				}
				k := uint32(j - lo)
				owners[j] = Owner{Lo: k * ctas / w, Hi: (k + 1) * ctas / w}
			}
			lo = hi
		}
	default:
		return nil, fmt.Errorf("selection policy %s is not one this package defines", p)
	}
	for i, o := range owners {
		if o.Count() == 0 {
			return nil, fmt.Errorf("task %q was dealt no CTA", tasks[i].ID)
		}
	}
	return owners, nil
}
