package cudair

import (
	"fmt"
	"reflect"
	"strings"
)

// DAGEmitter opts into context-free fan-out emission. Each ordered dependency
// takes the parent's complete packed input vector. Parent and child input and
// parameter types must match; compute types may differ. DAGKey identifies the
// immutable computation; DAGSymbol is its exact valid C++ function identifier.
// EmitForwardWithDependencies must call every supplied child once, in order,
// using input, params and the child's FwdCtxArg. Bodies use EmitFromBody's
// generated signature and the existing read-only write contract. Shared keys
// deduplicate definitions, not execution. Any nonzero forward context refuses.
//
// The initial contract admits at most 256 unique nodes, 4096 traversal visits,
// width 1048576, and 4 MiB of combined source. Keys are at most 4096 bytes;
// symbols are at most 256 ASCII identifier bytes and cannot be C++ keywords,
// reserved double-underscore names, generated parameter names, or the math
// intrinsics expf/erff/sqrtf/rsqrtf and helper lane_id. Declared
// shared memory must be nonnegative and at most 4 MiB; device limits still
// require separate validation. EmitTree's name must equal the root DAGSymbol.
// Emitters must remain immutable throughout both validation and emission.
type DAGEmitter interface {
	Emitter
	DAGKey() string
	DAGSymbol() string
	Dependencies() []DAGEmitter
	EmitForwardWithDependencies(name string, dependencies []ChildSymbol) (string, error)
}

const (
	dagMaxNodes  = 256
	dagMaxVisits = 4096
	dagMaxWidth  = 1 << 20
	dagMaxSource = 4 << 20
	dagMaxName   = 256
)

type dagDescriptor struct {
	symbol                          string
	in, out, padded, shmem          int
	aligned                         bool
	inType, paramsType, computeType string
	children                        string
}

type dagDefinition struct {
	descriptor dagDescriptor
	source     string
}

func nilDAG(e DAGEmitter) bool {
	if e == nil {
		return true
	}
	v := reflect.ValueOf(e)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}

func dagIdentifier(s string) bool {
	if s == "" || len(s) > dagMaxName {
		return false
	}
	for i, c := range s {
		if !(c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || i > 0 && c >= '0' && c <= '9') {
			return false
		}
	}
	if strings.HasPrefix(s, "__") || s == "input" || s == "params" || s == "fwd_ctx" || s == "tvec" || s == "expf" || s == "erff" || s == "sqrtf" || s == "rsqrtf" || s == "lane_id" {
		return false
	}
	for _, word := range strings.Fields("alignas alignof and and_eq asm auto bitand bitor bool break case catch char char8_t char16_t char32_t class compl concept const consteval constexpr constinit const_cast continue co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit export extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq private protected public register reinterpret_cast requires return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef typeid typename union unsigned using virtual void volatile wchar_t while xor xor_eq") {
		if s == word {
			return false
		}
	}
	return true
}

func emitDAG(root DAGEmitter, name string) (Unit, error) {
	if nilDAG(root) {
		return Unit{}, fmt.Errorf("cudair: nil DAG root")
	}
	if name != root.DAGSymbol() {
		return Unit{}, fmt.Errorf("cudair: DAG root name must match declared symbol")
	}
	definitions := make(map[string]dagDefinition)
	active := make(map[string]bool)
	symbolKeys := make(map[string]string)
	visits, totalBytes := 0, 0
	var unit Unit
	var sources []string
	emitting := false
	var visit func(DAGEmitter, int) (string, error)
	visit = func(node DAGEmitter, depth int) (string, error) {
		visits++
		if visits > dagMaxVisits || depth > dagMaxNodes {
			return "", fmt.Errorf("cudair: DAG traversal limit reached")
		}
		if nilDAG(node) {
			return "", fmt.Errorf("cudair: nil DAG dependency")
		}
		key, symbol := node.DAGKey(), node.DAGSymbol()
		if key == "" || len(key) > 4096 || !dagIdentifier(symbol) {
			return "", fmt.Errorf("cudair: invalid DAG key or symbol")
		}
		if active[key] {
			return "", fmt.Errorf("cudair: DAG cycle at %s", symbol)
		}
		if prior, ok := symbolKeys[symbol]; ok && prior != key {
			return "", fmt.Errorf("cudair: DAG symbol %s has conflicting keys", symbol)
		}
		if _, ok := symbolKeys[symbol]; !ok {
			if len(symbolKeys) >= dagMaxNodes {
				return "", fmt.Errorf("cudair: DAG node limit reached")
			}
			symbolKeys[symbol] = key
		}
		ctx, err := node.FwdCtxBytes()
		if err != nil {
			return "", fmt.Errorf("cudair: DAG context: %w", err)
		}
		if ctx != 0 {
			return "", fmt.Errorf("cudair: DAG requires zero forward context")
		}
		in, out, padded := node.InputWidth(), node.OutputWidth(), node.PaddedOutputWidth()
		if in <= 0 || out <= 0 || in > dagMaxWidth || out > dagMaxWidth || padded < out || padded > dagMaxWidth {
			return "", fmt.Errorf("cudair: invalid or oversized DAG widths")
		}
		shmem := 0
		if p, ok := node.(ShmemProvider); ok {
			shmem = p.ShmemBytes()
			if shmem < 0 || shmem > dagMaxSource {
				return "", fmt.Errorf("cudair: invalid DAG shared memory declaration")
			}
		}
		deps := node.Dependencies()
		if len(deps) > dagMaxNodes {
			return "", fmt.Errorf("cudair: DAG dependency limit reached")
		}
		active[key] = true
		children := make([]ChildSymbol, 0, len(deps))
		childKeys := make([]string, 0, len(deps))
		parentInput, parentParams, _ := getTypes(node)
		for _, child := range deps {
			if nilDAG(child) {
				return "", fmt.Errorf("cudair: nil DAG dependency")
			}
			if child.InputWidth() != in {
				return "", fmt.Errorf("cudair: DAG dependency packed-input width mismatch")
			}
			childInput, childParams, _ := getTypes(child)
			if childInput != parentInput || childParams != parentParams {
				return "", fmt.Errorf("cudair: DAG dependency input or parameter ABI differs")
			}
			s, err := visit(child, depth+1)
			if err != nil {
				return "", err
			}
			children = append(children, ChildSymbol{Name: s})
			childKeys = append(childKeys, child.DAGKey())
		}
		delete(active, key)
		it, pt, ct := getTypes(node)
		desc := dagDescriptor{symbol: symbol, in: in, out: out, padded: padded, shmem: shmem, aligned: node.FwdCtxAlignedPerElement(), inType: it, paramsType: pt, computeType: ct, children: fmt.Sprintf("%q", childKeys)}
		if !emitting {
			if prior, ok := definitions[key]; ok && prior.descriptor != desc {
				return "", fmt.Errorf("cudair: DAG key has inconsistent descriptor")
			}
			definitions[key] = dagDefinition{descriptor: desc}
			return symbol, nil
		}
		source, err := node.EmitForwardWithDependencies(symbol, append([]ChildSymbol(nil), children...))
		if err != nil {
			return "", fmt.Errorf("cudair: DAG emit %s: %w", symbol, err)
		}
		if len(source) > dagMaxSource {
			return "", fmt.Errorf("cudair: DAG function source limit reached")
		}
		if actual, ok := emittedFunctionName(source); !ok || actual != symbol {
			return "", fmt.Errorf("cudair: DAG emitted symbol differs")
		}
		if err := validateEmittedContextUse(node, source, children, ""); err != nil {
			return "", err
		}
		if err := validateDAGCalls(source, children); err != nil {
			return "", err
		}
		if prior, ok := definitions[key]; ok {
			if prior.descriptor != desc || prior.source != source {
				return "", fmt.Errorf("cudair: DAG key has inconsistent descriptor or source")
			}
			return symbol, nil
		}
		if len(source) > dagMaxSource-totalBytes-2 {
			return "", fmt.Errorf("cudair: DAG source limit reached")
		}
		totalBytes += len(source) + 2
		definitions[key] = dagDefinition{desc, source}
		sources = append(sources, source)
		unit.Symbols = append(unit.Symbols, symbol)
		if shmem > unit.ShmemBytes {
			unit.ShmemBytes = shmem
		}
		return symbol, nil
	}
	if _, err := visit(root, 1); err != nil {
		return Unit{}, err
	}
	definitions = make(map[string]dagDefinition)
	active = make(map[string]bool)
	symbolKeys = make(map[string]string)
	visits = 0
	emitting = true
	if _, err := visit(root, 1); err != nil {
		return Unit{}, err
	}
	unit.Source = strings.Join(sources, "\n\n")
	return unit, nil
}

func validateDAGCalls(source string, children []ChildSymbol) error {
	tokens, err := tokenizeCUDA(source)
	if err != nil {
		return err
	}
	allowed := make(map[string]bool)
	for _, c := range children {
		allowed[c.Name] = true
	}
	next := 0
	for i := 0; i+1 < len(tokens); i++ {
		if !allowed[tokens[i].text] || tokens[i+1].text != "(" {
			continue
		}
		_, args, ok := callArguments(tokens, i+1)
		if !ok || len(args) != 3 || next >= len(children) || tokens[i].text != children[next].Name {
			return fmt.Errorf("cudair: DAG dependency calls differ from declared order")
		}
		for j, want := range []string{"input", "params", children[next].FwdCtxArg()} {
			if tokenText(source, tokens[args[j][0]:args[j][1]]) != want {
				return fmt.Errorf("cudair: DAG dependency does not receive complete input and read-only parameters")
			}
		}
		next++
	}
	if next != len(children) {
		return fmt.Errorf("cudair: DAG dependency omitted from emitted body")
	}
	return nil
}
