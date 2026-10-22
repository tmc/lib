package cudashape

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrUnboundSymbol indicates that a symbolic dimension is missing a concrete binding.
	ErrUnboundSymbol = errors.New("cudashape: unbound symbol")

	// ErrContradictoryBinding indicates an attempt to bind a symbol to conflicting values.
	ErrContradictoryBinding = errors.New("cudashape: contradictory symbol binding")

	// ErrConstraintViolation indicates that dimension bindings violate an explicit constraint.
	ErrConstraintViolation = errors.New("cudashape: constraint violation")
)

// Dim represents a dimension that is either a fixed constant extent or a named symbol.
type Dim struct {
	name     string
	constant int
	isSymbol bool
}

// NamedDim constructs a symbolic dimension with the given identifier.
// It returns an error if name is empty.
func NamedDim(name string) (Dim, error) {
	if name == "" {
		return Dim{}, errors.New("cudashape: empty symbol name")
	}
	return Dim{name: name, isSymbol: true}, nil
}

// ConstDim constructs a fixed concrete dimension extent.
func ConstDim(v int) Dim {
	return Dim{constant: v, isSymbol: false}
}

// IsSymbol reports whether the dimension is symbolic.
func (d Dim) IsSymbol() bool {
	return d.isSymbol
}

// Name returns the symbol name, or empty string if constant.
func (d Dim) Name() string {
	return d.name
}

// Value returns the constant extent value if not symbolic.
func (d Dim) Value() int {
	return d.constant
}

// String returns a human-readable representation.
func (d Dim) String() string {
	if d.isSymbol {
		return d.name
	}
	return fmt.Sprintf("%d", d.constant)
}

// SymbolicShape represents a tensor shape with named symbols and/or constant extents.
type SymbolicShape struct {
	dims []Dim
}

// NewSymbolicShape creates a new SymbolicShape from dimensions.
func NewSymbolicShape(dims ...Dim) SymbolicShape {
	return SymbolicShape{dims: append([]Dim(nil), dims...)}
}

// Dims returns a copy of the symbolic dimensions.
func (ss SymbolicShape) Dims() []Dim {
	if len(ss.dims) == 0 {
		return nil
	}
	return append([]Dim(nil), ss.dims...)
}

// Rank returns the number of dimensions.
func (ss SymbolicShape) Rank() int {
	return len(ss.dims)
}

// String returns a string representation such as "[B, S, 128]".
func (ss SymbolicShape) String() string {
	if len(ss.dims) == 0 {
		return "[]"
	}
	var b strings.Builder
	b.WriteString("[")
	for i, d := range ss.dims {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(d.String())
	}
	b.WriteString("]")
	return b.String()
}

// Constraint defines an invariant over symbolic bindings.
type Constraint interface {
	Check(env *Environment) error
	String() string
}

type equalSymbolsConstraint struct {
	symA string
	symB string
}

func (c equalSymbolsConstraint) Check(env *Environment) error {
	vA, okA := env.Value(c.symA)
	vB, okB := env.Value(c.symB)
	if okA && okB && vA != vB {
		return fmt.Errorf("equality constraint violated: %s (%d) != %s (%d)", c.symA, vA, c.symB, vB)
	}
	return nil
}

func (c equalSymbolsConstraint) String() string {
	return fmt.Sprintf("%s == %s", c.symA, c.symB)
}

// EqualSymbols requires two named symbols to have identical values when bound.
func EqualSymbols(symA, symB string) Constraint {
	return equalSymbolsConstraint{symA: symA, symB: symB}
}

type equalConstConstraint struct {
	sym string
	val int
}

func (c equalConstConstraint) Check(env *Environment) error {
	v, ok := env.Value(c.sym)
	if ok && v != c.val {
		return fmt.Errorf("constant constraint violated: %s (%d) != expected %d", c.sym, v, c.val)
	}
	return nil
}

func (c equalConstConstraint) String() string {
	return fmt.Sprintf("%s == %d", c.sym, c.val)
}

// EqualConst requires a named symbol to equal a specified integer constant.
func EqualConst(sym string, val int) Constraint {
	return equalConstConstraint{sym: sym, val: val}
}

type divisibleByConstraint struct {
	sym     string
	divisor int
}

func (c divisibleByConstraint) Check(env *Environment) error {
	if c.divisor <= 0 {
		return errors.New("divisor must be positive")
	}
	v, ok := env.Value(c.sym)
	if ok && v%c.divisor != 0 {
		return fmt.Errorf("divisibility constraint violated: %s (%d) not divisible by %d", c.sym, v, c.divisor)
	}
	return nil
}

func (c divisibleByConstraint) String() string {
	return fmt.Sprintf("%s %% %d == 0", c.sym, c.divisor)
}

// DivisibleBy requires a named symbol to be evenly divisible by divisor.
func DivisibleBy(sym string, divisor int) Constraint {
	return divisibleByConstraint{sym: sym, divisor: divisor}
}

type rangeBoundConstraint struct {
	sym    string
	minVal int
	maxVal int
}

func (c rangeBoundConstraint) Check(env *Environment) error {
	v, ok := env.Value(c.sym)
	if ok {
		if v < c.minVal || v > c.maxVal {
			return fmt.Errorf("range constraint violated: %s (%d) outside [%d, %d]", c.sym, v, c.minVal, c.maxVal)
		}
	}
	return nil
}

func (c rangeBoundConstraint) String() string {
	return fmt.Sprintf("%d <= %s <= %d", c.minVal, c.sym, c.maxVal)
}

// RangeBound requires a named symbol's value to lie within [minVal, maxVal].
func RangeBound(sym string, minVal, maxVal int) Constraint {
	return rangeBoundConstraint{sym: sym, minVal: minVal, maxVal: maxVal}
}

// Environment manages dimension bindings and verifies registered constraints.
type Environment struct {
	bindings    map[string]int
	constraints []Constraint
}

// NewEnvironment creates an empty symbolic binding environment.
func NewEnvironment() *Environment {
	return &Environment{
		bindings: make(map[string]int),
	}
}

// Bind assigns a concrete extent value to a named symbol.
// If the symbol is already bound to a different value, ErrContradictoryBinding is returned.
// If the binding violates any registered constraint, ErrConstraintViolation is returned and the
// environment remains unmodified.
func (e *Environment) Bind(name string, val int) error {
	if name == "" {
		return errors.New("cudashape: empty symbol name")
	}
	if val < 0 {
		return fmt.Errorf("%w: symbol %s cannot be negative (%d)", ErrNegativeDimension, name, val)
	}
	if old, exists := e.bindings[name]; exists {
		if old != val {
			return fmt.Errorf("%w: symbol %q already bound to %d, rejected new binding %d",
				ErrContradictoryBinding, name, old, val)
		}
		return nil
	}

	// Tentatively apply binding
	e.bindings[name] = val
	for _, c := range e.constraints {
		if err := c.Check(e); err != nil {
			delete(e.bindings, name)
			return fmt.Errorf("%w: %s", ErrConstraintViolation, err.Error())
		}
	}
	return nil
}

// AddConstraint registers a constraint and checks it against current bindings.
// Returns ErrConstraintViolation if existing bindings violate the constraint.
func (e *Environment) AddConstraint(c Constraint) error {
	if err := c.Check(e); err != nil {
		return fmt.Errorf("%w: %s", ErrConstraintViolation, err.Error())
	}
	e.constraints = append(e.constraints, c)
	return nil
}

// Value looks up the concrete extent bound to name.
func (e *Environment) Value(name string) (int, bool) {
	v, ok := e.bindings[name]
	return v, ok
}

// Bindings returns a copy of all current bindings.
func (e *Environment) Bindings() map[string]int {
	m := make(map[string]int, len(e.bindings))
	for k, v := range e.bindings {
		m[k] = v
	}
	return m
}

// Bind evaluates the SymbolicShape against the environment, producing a concrete Shape.
// Returns ErrUnboundSymbol if any named dimension lacks a binding.
// Returns ErrConstraintViolation if any constraint is not satisfied.
func (ss SymbolicShape) Bind(env *Environment) (Shape, error) {
	for _, c := range env.constraints {
		if err := c.Check(env); err != nil {
			return Shape{}, fmt.Errorf("%w: %s", ErrConstraintViolation, err.Error())
		}
	}
	concreteDims := make([]int, len(ss.dims))
	for i, d := range ss.dims {
		if d.isSymbol {
			val, ok := env.Value(d.name)
			if !ok {
				return Shape{}, fmt.Errorf("%w: symbol %q at axis %d", ErrUnboundSymbol, d.name, i)
			}
			concreteDims[i] = val
		} else {
			concreteDims[i] = d.constant
		}
	}
	return New(concreteDims...)
}
