package egraphcore

import (
	"fmt"
	"sort"
)

// ChoiceMaterializer reuses dense traversal scratch for one fixed set of
// classes. Class IDs and child IDs must be canonical. It is not safe for
// concurrent or reentrant use.
type ChoiceMaterializer[ID ~uint32, Node, Value any] struct {
	classAt   map[ID]int
	state     []uint8
	values    []Value
	stack     []choiceMaterializeFrame[ID, Node]
	reachable []ID
}

type choiceMaterializeFrame[ID ~uint32, Node any] struct {
	class    ID
	node     Node
	children []ID
	next     int
}

// NewChoiceMaterializer creates reusable traversal scratch for classes. The
// class list is copied and must contain each class at most once. Its IDs must
// use the same canonical form as the IDs passed to Materialize.
func NewChoiceMaterializer[ID ~uint32, Node, Value any](classes []ID) (*ChoiceMaterializer[ID, Node, Value], error) {
	classAt := make(map[ID]int, len(classes))
	for index, class := range classes {
		if _, exists := classAt[class]; exists {
			return nil, fmt.Errorf("%w: duplicate materializer class %d", ErrInvalidChoiceGraph, class)
		}
		classAt[class] = index
	}
	return &ChoiceMaterializer[ID, Node, Value]{
		classAt:   classAt,
		state:     make([]uint8, len(classes)),
		values:    make([]Value, len(classes)),
		stack:     make([]choiceMaterializeFrame[ID, Node], 0, len(classes)),
		reachable: make([]ID, 0, len(classes)),
	}, nil
}

// ReachableClasses returns selected classes in ascending ID order. The
// returned slice is owned by the materializer and remains valid only until
// the next ReachableClasses call. It shares traversal scratch with Materialize
// and is not safe for concurrent or reentrant use. alternatives and children
// must return stable results for the duration of the call.
func (m *ChoiceMaterializer[ID, Node, Value]) ReachableClasses(
	root ID,
	choice map[ID]int,
	alternatives func(ID) []Node,
	children func(Node) []ID,
) ([]ID, error) {
	if m == nil || m.classAt == nil || alternatives == nil || children == nil {
		return nil, ErrInvalidChoiceGraph
	}
	clear(m.state)
	clear(m.stack)
	m.stack = m.stack[:0]
	m.reachable = m.reachable[:0]
	defer func() {
		clear(m.state)
		clear(m.stack)
		m.stack = m.stack[:0]
	}()

	newFrame := func(class ID) (choiceMaterializeFrame[ID, Node], int, error) {
		index, known := m.classAt[class]
		if !known {
			return choiceMaterializeFrame[ID, Node]{}, 0, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, class)
		}
		nodes := alternatives(class)
		alternative, ok := choice[class]
		if !ok {
			return choiceMaterializeFrame[ID, Node]{}, 0, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, class)
		}
		if alternative < 0 || alternative >= len(nodes) {
			return choiceMaterializeFrame[ID, Node]{}, 0, fmt.Errorf("%w: alternative %d out of range for class %d", ErrInvalidChoiceGraph, alternative, class)
		}
		node := nodes[alternative]
		return choiceMaterializeFrame[ID, Node]{
			class: class, node: node, children: children(node),
		}, index, nil
	}

	rootFrame, rootIndex, err := newFrame(root)
	if err != nil {
		return nil, err
	}
	m.state[rootIndex] = 1
	m.stack = append(m.stack, rootFrame)
	for len(m.stack) > 0 {
		last := len(m.stack) - 1
		current := &m.stack[last]
		if current.next < len(current.children) {
			child := current.children[current.next]
			childIndex, known := m.classAt[child]
			if !known {
				return nil, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, child)
			}
			switch m.state[childIndex] {
			case 1:
				return nil, fmt.Errorf("%w: cycle at class %d", ErrInvalidChoiceGraph, child)
			case 2:
				current.next++
			default:
				childFrame, index, err := newFrame(child)
				if err != nil {
					return nil, err
				}
				m.state[index] = 1
				m.stack = append(m.stack, childFrame)
			}
			continue
		}
		m.state[m.classAt[current.class]] = 2
		m.reachable = append(m.reachable, current.class)
		var zero choiceMaterializeFrame[ID, Node]
		m.stack[last] = zero
		m.stack = m.stack[:last]
		if len(m.stack) > 0 {
			m.stack[len(m.stack)-1].next++
		}
	}
	sort.Slice(m.reachable, func(i, j int) bool { return m.reachable[i] < m.reachable[j] })
	return m.reachable, nil
}

// Materialize builds the selected root value in postorder. It validates
// reachable choices, rejects cycles, and reuses each completed class value
// within this result so shared children remain shared. On success, build is
// called once per reachable class, with child values in node child order.
// Scratch is cleared after success or error; no values are retained between
// calls. The choice space and callbacks must remain stable during the call;
// alternatives and children must return results whose contents and backing
// storage remain valid and unchanged until the call ends.
func (m *ChoiceMaterializer[ID, Node, Value]) Materialize(
	root ID,
	choice map[ID]int,
	alternatives func(ID) []Node,
	children func(Node) []ID,
	build func(class ID, node Node, children []Value) (Value, error),
) (result Value, err error) {
	if m == nil || m.classAt == nil || alternatives == nil || children == nil || build == nil {
		return result, ErrInvalidChoiceGraph
	}
	clear(m.state)
	clear(m.values)
	clear(m.stack)
	m.stack = m.stack[:0]
	defer func() {
		clear(m.state)
		clear(m.values)
		clear(m.stack)
		m.stack = m.stack[:0]
	}()

	newFrame := func(class ID) (choiceMaterializeFrame[ID, Node], int, error) {
		index, known := m.classAt[class]
		if !known {
			return choiceMaterializeFrame[ID, Node]{}, 0, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, class)
		}
		nodes := alternatives(class)
		alternative, ok := choice[class]
		if !ok {
			return choiceMaterializeFrame[ID, Node]{}, 0, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, class)
		}
		if alternative < 0 || alternative >= len(nodes) {
			return choiceMaterializeFrame[ID, Node]{}, 0, fmt.Errorf("%w: alternative %d out of range for class %d", ErrInvalidChoiceGraph, alternative, class)
		}
		node := nodes[alternative]
		return choiceMaterializeFrame[ID, Node]{
			class:    class,
			node:     node,
			children: children(node),
		}, index, nil
	}

	rootFrame, rootIndex, err := newFrame(root)
	if err != nil {
		return result, err
	}
	m.state[rootIndex] = 1
	m.stack = append(m.stack, rootFrame)
	for len(m.stack) != 0 {
		last := len(m.stack) - 1
		current := &m.stack[last]
		if current.next < len(current.children) {
			child := current.children[current.next]
			childIndex, known := m.classAt[child]
			if !known {
				return result, fmt.Errorf("%w: missing choice for class %d", ErrInvalidChoiceGraph, child)
			}
			switch m.state[childIndex] {
			case 1:
				return result, fmt.Errorf("%w: cycle at class %d", ErrInvalidChoiceGraph, child)
			case 2:
				current.next++
			default:
				childFrame, index, err := newFrame(child)
				if err != nil {
					return result, err
				}
				m.state[index] = 1
				m.stack = append(m.stack, childFrame)
			}
			continue
		}
		childValues := make([]Value, len(current.children))
		for index, child := range current.children {
			childValues[index] = m.values[m.classAt[child]]
		}
		value, err := build(current.class, current.node, childValues)
		if err != nil {
			return result, fmt.Errorf("egraphcore: build selected class %d: %w", current.class, err)
		}
		classIndex := m.classAt[current.class]
		m.values[classIndex] = value
		m.state[classIndex] = 2
		var zeroFrame choiceMaterializeFrame[ID, Node]
		m.stack[last] = zeroFrame
		m.stack = m.stack[:last]
		if len(m.stack) == 0 {
			return value, nil
		}
		parent := &m.stack[len(m.stack)-1]
		parent.next++
	}
	return result, fmt.Errorf("%w: failed to materialize root class %d", ErrInvalidChoiceGraph, root)
}
