package egraphcore

import (
	"errors"
	"fmt"
	"math"
	"time"
)

// Schedule describes an ordered plan for running backend-defined rules.
// Construct schedules with Run, Saturate, Sequence, and Repeat.
type Schedule[R any] struct {
	kind     scheduleKind
	name     string
	rules    []R
	children []*Schedule[R]
	count    int
}

type scheduleKind uint8

const (
	scheduleRun scheduleKind = iota + 1
	scheduleSaturate
	scheduleSequence
	scheduleRepeat
)

var (
	// ErrInvalidSchedule reports a malformed schedule or missing callback.
	ErrInvalidSchedule = errors.New("egraphcore: invalid schedule")
	// ErrScheduleCounterOverflow reports an unrepresentable run or change count.
	ErrScheduleCounterOverflow = errors.New("egraphcore: schedule counter overflow")
)

// Run creates a schedule node that executes rules as one ordered rule group.
// name identifies the group in execution paths and must be non-empty.
func Run[R any](name string, rules ...R) *Schedule[R] {
	return &Schedule[R]{kind: scheduleRun, name: name, rules: append([]R(nil), rules...)}
}

// Saturate creates a node that repeats child until a full child execution
// reports no changes, the callback halts, or the execution run limit is met.
func Saturate[R any](child *Schedule[R]) *Schedule[R] {
	return SaturateNamed("", child)
}

// SaturateNamed creates a fixed-point node with a stable execution name.
func SaturateNamed[R any](name string, child *Schedule[R]) *Schedule[R] {
	return &Schedule[R]{kind: scheduleSaturate, name: name, children: []*Schedule[R]{child}}
}

// Sequence creates a node that executes children in order.
func Sequence[R any](children ...*Schedule[R]) *Schedule[R] {
	return SequenceNamed("", children...)
}

// SequenceNamed creates an ordered sequence with a stable execution name.
func SequenceNamed[R any](name string, children ...*Schedule[R]) *Schedule[R] {
	return &Schedule[R]{kind: scheduleSequence, name: name, children: append([]*Schedule[R](nil), children...)}
}

// Repeat creates a node that executes child at most count times. It stops
// early when one complete child execution reports no changes.
func Repeat[R any](count int, child *Schedule[R]) *Schedule[R] {
	return RepeatNamed("", count, child)
}

// RepeatNamed creates a bounded repeat node with a stable execution name.
func RepeatNamed[R any](name string, count int, child *Schedule[R]) *Schedule[R] {
	return &Schedule[R]{kind: scheduleRepeat, name: name, count: count, children: []*Schedule[R]{child}}
}

// ScheduleRuleValidator validates one backend-defined rule.
type ScheduleRuleValidator[R any] func(R) error

// ValidateSchedule checks schedule structure and validates every rule. It
// reports cycles and nil nodes before execution begins.
func ValidateSchedule[R any](schedule *Schedule[R], validateRule ScheduleRuleValidator[R]) error {
	if schedule == nil || validateRule == nil {
		return ErrInvalidSchedule
	}
	visiting := make(map[*Schedule[R]]bool)
	validated := make(map[*Schedule[R]]bool)
	var visit func(*Schedule[R], string) error
	visit = func(node *Schedule[R], path string) error {
		if node == nil {
			return fmt.Errorf("%w: nil node at %s", ErrInvalidSchedule, path)
		}
		if visiting[node] {
			return fmt.Errorf("%w: cycle at %s", ErrInvalidSchedule, path)
		}
		if validated[node] {
			return nil
		}
		visiting[node] = true
		defer delete(visiting, node)
		switch node.kind {
		case scheduleRun:
			if node.name == "" {
				return fmt.Errorf("%w: empty rule group name at %s", ErrInvalidSchedule, path)
			}
			for i, rule := range node.rules {
				if err := validateRule(rule); err != nil {
					return fmt.Errorf("%s/run(%s)/rule[%d]: %w", path, node.name, i, err)
				}
			}
		case scheduleSaturate, scheduleSequence, scheduleRepeat:
			if node.kind == scheduleRepeat && node.count < 0 {
				return fmt.Errorf("%w: repeat count must be non-negative at %s", ErrInvalidSchedule, path)
			}
			if node.kind != scheduleSequence && len(node.children) != 1 {
				return fmt.Errorf("%w: expected one child at %s", ErrInvalidSchedule, path)
			}
			for i, child := range node.children {
				if node.kind == scheduleSequence {
					if err := visit(child, fmt.Sprintf("%s/sequence[%d]", path, i)); err != nil {
						return err
					}
				} else if err := visit(child, path+"/child"); err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("%w: unknown node kind at %s", ErrInvalidSchedule, path)
		}
		validated[node] = true
		return nil
	}
	return visit(schedule, "schedule")
}

// ScheduleRun executes one ordered rule group as a phase. It returns the
// phase's change count, whether execution should halt, and any error.
type ScheduleRun[R any] func(name string, rules []R) (changes int, halt bool, err error)

// ScheduleResult reports executions and changes. RunLimitReached is true when
// the maximum number of rule callbacks was reached before the plan completed.
type ScheduleResult struct {
	Runs            int
	Changes         int
	Halted          bool
	RunLimitReached bool
	Reports         []ScheduleRunReport
}

// ScheduleRunReport records one rule-group callback, including attempts that
// return an error. Path identifies the nested schedule location.
type ScheduleRunReport struct {
	Path     string
	Changes  int
	Duration time.Duration
}

// ExecuteSchedule validates and executes schedule in order. maxRuns must be
// positive and bounds the number of rule callback invocations. Run names are
// deterministic paths such as schedule/sequence[1]/repeat[0]/run(rewrite).
func ExecuteSchedule[R any](schedule *Schedule[R], maxRuns int, validateRule ScheduleRuleValidator[R], run ScheduleRun[R]) (ScheduleResult, error) {
	if maxRuns <= 0 || run == nil {
		return ScheduleResult{}, ErrInvalidSchedule
	}
	if err := ValidateSchedule(schedule, validateRule); err != nil {
		return ScheduleResult{}, err
	}
	result := ScheduleResult{}
	var execute func(*Schedule[R], string) (int, bool, error)
	execute = func(node *Schedule[R], path string) (int, bool, error) {
		switch node.kind {
		case scheduleRun:
			if result.Runs >= maxRuns {
				result.RunLimitReached = true
				return 0, true, nil
			}
			name := path + "/run(" + node.name + ")"
			if result.Runs == math.MaxInt {
				return 0, true, ErrScheduleCounterOverflow
			}
			result.Runs++
			started := time.Now()
			changes, halt, err := run(name, append([]R(nil), node.rules...))
			result.Reports = append(result.Reports, ScheduleRunReport{
				Path: name, Changes: changes, Duration: time.Since(started),
			})
			if err != nil {
				return 0, true, fmt.Errorf("%s: %w", name, err)
			}
			if changes < 0 {
				return 0, true, fmt.Errorf("%w: negative change count at %s", ErrInvalidSchedule, name)
			}
			if changes > math.MaxInt-result.Changes {
				return 0, true, ErrScheduleCounterOverflow
			}
			result.Changes += changes
			if halt {
				result.Halted = true
			}
			return changes, halt, nil
		case scheduleSequence:
			var changes int
			sequencePath := path
			if node.name != "" {
				sequencePath += "/" + node.name
			}
			for i, child := range node.children {
				n, halt, err := execute(child, fmt.Sprintf("%s/sequence[%d]", sequencePath, i))
				if n > math.MaxInt-changes {
					return 0, true, ErrScheduleCounterOverflow
				}
				changes += n
				if err != nil || halt {
					return changes, halt, err
				}
			}
			return changes, false, nil
		case scheduleSaturate, scheduleRepeat:
			limit := node.count
			if node.kind == scheduleSaturate {
				limit = maxRuns
			}
			var total int
			controlPath := path
			if node.name != "" {
				controlPath += "/" + node.name
			}
			for i := 0; i < limit; i++ {
				iteration := "saturate"
				if node.kind == scheduleRepeat {
					iteration = "repeat"
				}
				n, halt, err := execute(node.children[0], fmt.Sprintf("%s/%s[%d]", controlPath, iteration, i))
				if n > math.MaxInt-total {
					return 0, true, ErrScheduleCounterOverflow
				}
				total += n
				if err != nil || halt {
					return total, halt, err
				}
				if n == 0 {
					return total, false, nil
				}
				if result.RunLimitReached {
					return total, true, nil
				}
				if node.kind == scheduleSaturate && result.Runs >= maxRuns {
					result.RunLimitReached = true
					return total, true, nil
				}
			}
			return total, false, nil
		default:
			return 0, true, ErrInvalidSchedule
		}
	}
	_, _, err := execute(schedule, "schedule")
	if err != nil {
		return result, err
	}
	return result, nil
}
