package egraph

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Presort identifies a parameterized built-in sort.
type Presort uint8

const (
	PresortVec Presort = iota + 1
	PresortSet
	PresortMap
	PresortMultiSet
)

func (p Presort) nameAndArity() (string, int, bool) {
	switch p {
	case PresortVec:
		return "Vec", 1, true
	case PresortSet:
		return "Set", 1, true
	case PresortMap:
		return "Map", 2, true
	case PresortMultiSet:
		return "MultiSet", 1, true
	default:
		return "", 0, false
	}
}

func builtinPresortArity(name string) (int, bool) {
	switch name {
	case "Vec", "Set", "MultiSet":
		return 1, true
	case "Map":
		return 2, true
	default:
		return 0, false
	}
}

func builtinSortName(name string) bool {
	switch name {
	case "i64", "f64", "bool", "String", "BigInt", "BigRat", "Unit":
		return true
	default:
		return false
	}
}

// ContainerSort returns the canonical spelling for a parameterized sort.
// Arguments may be atomic sort names or other ContainerSort results.
func ContainerSort(presort Presort, arguments ...string) (string, error) {
	name, arity, ok := presort.nameAndArity()
	if !ok {
		return "", fmt.Errorf("egraph: invalid presort %d", presort)
	}
	if len(arguments) != arity {
		return "", fmt.Errorf("egraph: presort %s requires %d sort arguments, got %d", name, arity, len(arguments))
	}
	for i, argument := range arguments {
		if err := validateSortExpression(argument); err != nil {
			return "", fmt.Errorf("egraph: presort %s argument %d: %w", name, i, err)
		}
	}
	var result strings.Builder
	result.WriteString(name)
	result.WriteByte('<')
	for i, argument := range arguments {
		if i != 0 {
			result.WriteByte(',')
		}
		result.WriteString(argument)
	}
	result.WriteByte('>')
	return result.String(), nil
}

func validateSortExpression(value string) error {
	if value == "" {
		return fmt.Errorf("sort argument is empty")
	}
	index := 0
	var parse func(int) error
	parse = func(depth int) error {
		if depth > 128 {
			return fmt.Errorf("sort expression nesting exceeds 128 levels")
		}
		start := index
		for index < len(value) {
			r, size := rune(value[index]), 1
			if r >= 0x80 {
				r, size = utf8.DecodeRuneInString(value[index:])
				if r == utf8.RuneError && size == 1 {
					return fmt.Errorf("invalid UTF-8 in sort expression %q", value)
				}
			}
			valid := unicode.IsLetter(r) || r == '_'
			if index != start {
				valid = valid || unicode.IsDigit(r) || r == '-' || r == '.'
			}
			if r == '<' || r == '>' || r == ',' || unicode.IsSpace(r) || !valid {
				break
			}
			index += size
		}
		if index == start {
			return fmt.Errorf("invalid sort expression %q", value)
		}
		name := value[start:index]
		if index == len(value) || value[index] != '<' {
			if _, isPresort := builtinPresortArity(name); isPresort {
				return fmt.Errorf("presort %s requires sort arguments", name)
			}
			return nil
		}
		index++
		arguments := 0
		for {
			if err := parse(depth + 1); err != nil {
				return err
			}
			arguments++
			if index >= len(value) {
				return fmt.Errorf("unterminated sort expression %q", value)
			}
			switch value[index] {
			case ',':
				index++
			case '>':
				index++
				if arity, isPresort := builtinPresortArity(name); isPresort && arguments != arity {
					return fmt.Errorf("presort %s requires %d sort arguments, got %d", name, arity, arguments)
				}
				return nil
			default:
				return fmt.Errorf("invalid sort expression %q", value)
			}
		}
	}
	if err := parse(0); err != nil {
		return err
	}
	if index != len(value) {
		return fmt.Errorf("invalid sort expression %q", value)
	}
	return nil
}

// OperatorSignature declares the result sort and ordered child sorts for an
// operator. An empty child sort accepts any sort; the result sort is required.
type OperatorSignature struct {
	Result   string
	Children []string
}

// Datatype declares an e-class sort and its constructor operators.
type Datatype struct {
	Name     string
	Variants []DatatypeVariant
}

// DatatypeVariant declares one constructor and the sorts of its fields.
type DatatypeVariant struct {
	Name   string
	Fields []string
}

// RegisterDatatype declares a datatype and all of its constructor operators.
// It enables strict operator and sort checking: nodes must have a declared
// result sort, and every operator with children must have a declared signature.
// Field sorts may name this datatype or already registered datatypes. Use
// RegisterDatatypes for mutually recursive declarations.
// Registration is atomic: invalid declarations leave the graph unchanged.
func (g *EGraph) RegisterDatatype(datatype Datatype) error {
	return g.RegisterDatatypes(datatype)
}

// RegisterDatatypes atomically declares datatypes and their constructor
// operators. Field sorts may refer to any datatype in the batch, allowing
// mutually recursive declarations. Every field and existing operator sort
// must resolve to a built-in sort, a declared datatype, or a parameterized
// built-in sort. After registration, graph nodes and table class columns must
// use declared sorts.
func (g *EGraph) RegisterDatatypes(datatypes ...Datatype) error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.signatures == nil || g.datatypes == nil || g.variantOwners == nil {
		return fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if g.store.Len() != 0 || g.dirty {
		return fmt.Errorf("egraph: datatypes must be registered before adding nodes")
	}
	if len(datatypes) == 0 {
		return nil
	}

	pending := make(map[string]Datatype, len(datatypes))
	for _, datatype := range datatypes {
		if err := validateSortExpression(datatype.Name); err != nil || strings.ContainsAny(datatype.Name, "<>,") {
			return fmt.Errorf("egraph: invalid datatype name %q", datatype.Name)
		}
		if _, isPresort := builtinPresortArity(datatype.Name); isPresort || builtinSortName(datatype.Name) {
			return fmt.Errorf("egraph: datatype name %q is built in", datatype.Name)
		}
		if _, duplicate := pending[datatype.Name]; duplicate {
			return fmt.Errorf("egraph: duplicate datatype %q in declaration batch", datatype.Name)
		}
		if previous, exists := g.datatypes[datatype.Name]; exists {
			if sameDatatype(previous, datatype) {
				continue
			}
			return fmt.Errorf("egraph: datatype %q is already registered", datatype.Name)
		}
		pending[datatype.Name] = datatype
	}

	declaredSorts := g.declaredSortNames()
	for name := range pending {
		declaredSorts[name] = true
	}
	for op, signatures := range g.signatures {
		for _, signature := range signatures {
			if err := validateDeclaredSortExpression(signature.Result, declaredSorts); err != nil {
				return fmt.Errorf("egraph: operator %q result sort: %w", op, err)
			}
			for i, child := range signature.Children {
				if child == "" {
					return fmt.Errorf("egraph: operator %q child sort %d is unconstrained", op, i)
				}
				if err := validateDeclaredSortExpression(child, declaredSorts); err != nil {
					return fmt.Errorf("egraph: operator %q child sort %d: %w", op, i, err)
				}
			}
		}
	}
	for name, relation := range g.relations {
		for i, column := range relation.schema.Columns {
			if err := validateDeclaredClassColumnWithSorts(column, fmt.Sprintf("relation %q column %d", name, i), declaredSorts); err != nil {
				return fmt.Errorf("egraph: %w", err)
			}
		}
	}
	for name, function := range g.functions {
		for i, typ := range function.schema.KeyTypes {
			if err := validateDeclaredClassColumnWithSorts(typ, fmt.Sprintf("function %q key %d", name, i), declaredSorts); err != nil {
				return fmt.Errorf("egraph: %w", err)
			}
		}
		if err := validateDeclaredClassColumnWithSorts(function.schema.ValueType, fmt.Sprintf("function %q value", name), declaredSorts); err != nil {
			return fmt.Errorf("egraph: %w", err)
		}
	}

	seenConstructors := make(map[string]string)
	for _, datatype := range datatypes {
		if _, alreadyRegistered := g.datatypes[datatype.Name]; alreadyRegistered {
			continue
		}
		for _, variant := range datatype.Variants {
			if err := validateSortExpression(variant.Name); err != nil || strings.ContainsAny(variant.Name, "<>,") {
				return fmt.Errorf("egraph: invalid constructor name %q for datatype %q", variant.Name, datatype.Name)
			}
			if _, isPresort := builtinPresortArity(variant.Name); isPresort || builtinSortName(variant.Name) {
				return fmt.Errorf("egraph: constructor name %q is built in", variant.Name)
			}
			if owner, duplicate := seenConstructors[variant.Name]; duplicate {
				return fmt.Errorf("egraph: constructor %q is declared by both %q and %q", variant.Name, owner, datatype.Name)
			}
			seenConstructors[variant.Name] = datatype.Name
			if owner, ok := g.variantOwners[variant.Name]; ok {
				return fmt.Errorf("egraph: constructor %q already belongs to datatype %q", variant.Name, owner)
			}
			if g.functions[variant.Name] != nil || g.relations[variant.Name] != nil {
				return fmt.Errorf("egraph: constructor %q is already registered as a table", variant.Name)
			}
			if len(g.signatures[variant.Name]) != 0 {
				return fmt.Errorf("egraph: constructor %q already has an operator signature", variant.Name)
			}
			for i, field := range variant.Fields {
				if err := validateDeclaredSortExpression(field, declaredSorts); err != nil {
					return fmt.Errorf("egraph: datatype %q constructor %q field %d: %w", datatype.Name, variant.Name, i, err)
				}
			}
		}
	}

	for _, datatype := range datatypes {
		if _, alreadyRegistered := g.datatypes[datatype.Name]; alreadyRegistered {
			continue
		}
		copyOfDatatype := Datatype{Name: datatype.Name, Variants: make([]DatatypeVariant, len(datatype.Variants))}
		for i, variant := range datatype.Variants {
			fields := append([]string(nil), variant.Fields...)
			copyOfDatatype.Variants[i] = DatatypeVariant{Name: variant.Name, Fields: fields}
		}
		for _, variant := range copyOfDatatype.Variants {
			g.signatures[variant.Name] = []OperatorSignature{{
				Result:   datatype.Name,
				Children: append([]string(nil), variant.Fields...),
			}}
			g.variantOwners[variant.Name] = datatype.Name
		}
		g.datatypes[datatype.Name] = copyOfDatatype
	}
	g.strictSignatures = true
	g.declaredSortsStrict = true
	return nil
}

func (g *EGraph) declaredSortNames() map[string]bool {
	declared := make(map[string]bool, len(g.datatypes)+7)
	for name := range g.datatypes {
		declared[name] = true
	}
	for _, name := range []string{"i64", "f64", "bool", "String", "BigInt", "BigRat", "Unit"} {
		declared[name] = true
	}
	return declared
}

func validateDeclaredClassColumn(g *EGraph, typ FunctionColumnType, where string) error {
	return validateDeclaredClassColumnWithSorts(typ, where, g.declaredSortNames())
}

func validateDeclaredClassColumnWithSorts(typ FunctionColumnType, where string, declared map[string]bool) error {
	if typ.Kind != FunctionClassColumn {
		return nil
	}
	if typ.Sort == "" {
		return fmt.Errorf("%s has an unconstrained class sort", where)
	}
	if err := validateDeclaredSortExpression(typ.Sort, declared); err != nil {
		return fmt.Errorf("%s: %w", where, err)
	}
	return nil
}

func validateDeclaredSortExpression(value string, declared map[string]bool) error {
	if err := validateSortExpression(value); err != nil {
		return err
	}
	index := 0
	var parse func(int) error
	parse = func(depth int) error {
		start := index
		for index < len(value) {
			r, size := rune(value[index]), 1
			if r >= 0x80 {
				r, size = utf8.DecodeRuneInString(value[index:])
			}
			valid := unicode.IsLetter(r) || r == '_'
			if index != start {
				valid = valid || unicode.IsDigit(r) || r == '-' || r == '.'
			}
			if !valid {
				break
			}
			index += size
		}
		name := value[start:index]
		if index >= len(value) || value[index] != '<' {
			if !declared[name] {
				return fmt.Errorf("sort %q is not declared", name)
			}
			return nil
		}
		arity, isPresort := builtinPresortArity(name)
		if !isPresort {
			return fmt.Errorf("sort %q is not a parameterized built-in sort", name)
		}
		index++
		arguments := 0
		for {
			arguments++
			if err := parse(depth + 1); err != nil {
				return err
			}
			switch value[index] {
			case ',':
				index++
			case '>':
				index++
				if arguments != arity {
					return fmt.Errorf("presort %s requires %d sort arguments, got %d", name, arity, arguments)
				}
				return nil
			}
		}
	}
	if err := parse(0); err != nil {
		return err
	}
	if index != len(value) {
		return fmt.Errorf("invalid sort expression %q", value)
	}
	return nil
}

func sameDatatype(a, b Datatype) bool {
	if a.Name != b.Name || len(a.Variants) != len(b.Variants) {
		return false
	}
	for i := range a.Variants {
		if a.Variants[i].Name != b.Variants[i].Name || len(a.Variants[i].Fields) != len(b.Variants[i].Fields) {
			return false
		}
		for j := range a.Variants[i].Fields {
			if a.Variants[i].Fields[j] != b.Variants[i].Fields[j] {
				return false
			}
		}
	}
	return true
}

// RequireOperatorSignatures makes Add reject operators with children unless
// RegisterOperator declared a matching signature. Unregistered leaves remain
// allowed for symbols and literals unless RegisterDatatype has enabled declared
// sort checking. Call it before adding any nodes.
func (g *EGraph) RequireOperatorSignatures() error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.signatures == nil {
		return fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if g.store.Len() != 0 || g.dirty {
		return fmt.Errorf("egraph: strict signatures must be enabled before adding nodes")
	}
	g.strictSignatures = true
	return nil
}

// RegisterOperator adds a typed signature for op. Signatures must be
// registered before adding any nodes, so existing classes cannot violate the
// new declaration. Multiple signatures may overload an operator by arity or
// child sorts or explicit result sorts.
func (g *EGraph) RegisterOperator(op string, signature OperatorSignature) error {
	if g == nil {
		return fmt.Errorf("egraph: nil graph")
	}
	if !g.store.Initialized() || g.signatures == nil {
		return fmt.Errorf("egraph: graph is not initialized; use New")
	}
	if op == "" || signature.Result == "" {
		return fmt.Errorf("egraph: operator and result sort must be non-empty")
	}
	if owner, ok := g.variantOwners[op]; ok {
		return fmt.Errorf("egraph: operator %q is the constructor of datatype %q", op, owner)
	}
	if datatype, ok := g.datatypes[signature.Result]; ok && len(datatype.Variants) == 0 {
		return fmt.Errorf("egraph: operator %q cannot produce empty datatype %q", op, signature.Result)
	}
	if err := validateSortExpression(signature.Result); err != nil {
		return fmt.Errorf("egraph: operator %q result sort: %w", op, err)
	}
	for i, child := range signature.Children {
		if child == "" {
			continue
		}
		if err := validateSortExpression(child); err != nil {
			return fmt.Errorf("egraph: operator %q child sort %d: %w", op, i, err)
		}
	}
	if g.declaredSortsStrict {
		declared := g.declaredSortNames()
		if err := validateDeclaredSortExpression(signature.Result, declared); err != nil {
			return fmt.Errorf("egraph: operator %q result sort: %w", op, err)
		}
		for i, child := range signature.Children {
			if child == "" {
				return fmt.Errorf("egraph: operator %q child sort %d is unconstrained", op, i)
			}
			if err := validateDeclaredSortExpression(child, declared); err != nil {
				return fmt.Errorf("egraph: operator %q child sort %d: %w", op, i, err)
			}
		}
	}
	if g.store.Len() != 0 || g.dirty {
		return fmt.Errorf("egraph: operator signatures must be registered before adding nodes")
	}
	for _, existing := range g.signatures[op] {
		if sameSignature(existing, signature) {
			return nil
		}
		// Nodes declare their result sort, so distinct results cannot
		// compete when validating the same typed node.
		if existing.Result != signature.Result {
			continue
		}
		if len(existing.Children) == len(signature.Children) {
			compatible := true
			for i := range existing.Children {
				if existing.Children[i] != "" && signature.Children[i] != "" && existing.Children[i] != signature.Children[i] {
					compatible = false
					break
				}
			}
			if compatible {
				return fmt.Errorf("egraph: ambiguous signatures for operator %q", op)
			}
		}
	}
	signature.Children = append([]string(nil), signature.Children...)
	g.signatures[op] = append(g.signatures[op], signature)
	return nil
}

func sameSignature(a, b OperatorSignature) bool {
	if a.Result != b.Result || len(a.Children) != len(b.Children) {
		return false
	}
	for i := range a.Children {
		if a.Children[i] != b.Children[i] {
			return false
		}
	}
	return true
}

func (g *EGraph) validateSignature(node ENode) error {
	if g.declaredSortsStrict {
		if node.Sort == "" {
			return fmt.Errorf("egraph: node %q has no result sort in a declared-sort graph", node.Op)
		}
		if err := validateDeclaredSortExpression(node.Sort, g.declaredSortNames()); err != nil {
			return fmt.Errorf("egraph: node %q result sort: %w", node.Op, err)
		}
	}
	if datatype, ok := g.datatypes[node.Sort]; ok && len(datatype.Variants) == 0 {
		return fmt.Errorf("egraph: cannot add a node of empty datatype %q", node.Sort)
	}
	signatures := g.signatures[node.Op]
	if len(signatures) == 0 {
		if datatype, declared := g.datatypes[node.Sort]; declared && len(datatype.Variants) != 0 {
			return fmt.Errorf("egraph: operator %q is not a constructor of datatype %q", node.Op, node.Sort)
		}
		if g.strictSignatures && len(node.Children) > 0 {
			return fmt.Errorf("egraph: operator %q has no registered signature", node.Op)
		}
		return nil
	}
	var arityMatched bool
	for _, signature := range signatures {
		if len(signature.Children) != len(node.Children) {
			continue
		}
		arityMatched = true
		if node.Sort != signature.Result {
			continue
		}
		matches := true
		for i, want := range signature.Children {
			if want == "" {
				continue
			}
			class := g.store.Class(g.Find(node.Children[i]))
			if class == nil || class.Sort != want {
				matches = false
				break
			}
		}
		if matches {
			return nil
		}
	}
	if !arityMatched {
		return fmt.Errorf("egraph: operator %q has no signature with arity %d", node.Op, len(node.Children))
	}
	return fmt.Errorf("egraph: node %q does not match a registered sort signature", node.Op)
}
