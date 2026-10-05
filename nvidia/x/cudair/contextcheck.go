package cudair

import (
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
	"strings"
)

// validateGeneratedContextBody checks only function text produced by the
// fixed cudair signatures. The tokenizer treats identifiers as whole tokens,
// so child names and context references cannot be confused by substrings.
func validateGeneratedContextBody(node Emitter, source string, children []ChildSymbol, preamble string) error {
	fail := func(reason string) error {
		return fmt.Errorf("%w: node %s: %s", ErrContextBoundsExceeded, NodeName(node), reason)
	}
	if strings.Contains(source, "\\\n") || strings.Contains(source, "\\\r\n") || strings.Contains(source, "##") {
		return fail("unsupported source splice")
	}
	for _, line := range strings.Split(source, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			return fail("preprocessor directive")
		}
	}

	name, ok := emittedFunctionName(source)
	if !ok {
		return fail("source does not use a generated signature")
	}
	backwardSource := strings.HasPrefix(source, "__device__ void ")
	var generated string
	var err error
	if backwardSource {
		backward, isBackward := node.(BackwardEmitter)
		if !isBackward {
			return fail("unexpected backward signature")
		}
		generated, err = EmitBackwardFromBody(backward, name, "")
	} else {
		generated, err = EmitFromBody(node, name, "")
	}
	if err != nil {
		return fail("cannot generate signature")
	}
	// The generated empty function is the exact framing. Only its empty body
	// may be replaced by node text.
	marker := "\n\n}"
	prefix := strings.TrimSuffix(generated, marker)
	if !strings.HasSuffix(generated, marker) || !strings.HasPrefix(source, prefix) || !strings.HasSuffix(source, "\n}") {
		return fail("source is not a generated signature and body")
	}
	body := strings.TrimSuffix(strings.TrimPrefix(source, prefix), "\n}")
	if strings.HasPrefix(body, "\n") {
		body = body[1:]
	}
	body = strings.TrimPrefix(body, preamble)
	ctxBytes, err := node.FwdCtxBytes()
	if err != nil {
		return fail("cannot read context extent")
	}
	if backwardSource {
		body = strings.TrimPrefix(body, backwardContextAccessor(ctxBytes))
	} else {
		body = strings.TrimPrefix(body, forwardContextAccessor(ctxBytes))
	}
	tokens, err := tokenizeCUDA(body)
	if err != nil {
		return fail("body cannot be tokenized")
	}
	if !balancedTokens(tokens) {
		return fail("body has unbalanced delimiters")
	}
	allowed := make(map[int]bool)
	childArgs := 3
	if backwardSource {
		childArgs = 5
	}
	for _, child := range children {
		for i := 0; i+1 < len(tokens); i++ {
			if tokens[i].text != child.Name || tokens[i+1].text != "(" || (i > 0 && tokens[i-1].text == "::") {
				continue
			}
			end, args, ok := callArguments(tokens, i+1)
			if !ok || len(args) != childArgs || tokenText(body, tokens[args[2][0]:args[2][1]]) != child.FwdCtxArg() {
				continue
			}
			for j := args[2][0]; j < args[2][1]; j++ {
				if tokens[j].text == "fwd_ctx" {
					allowed[j] = true
				}
			}
			_ = end
		}
	}
	for i, token := range tokens {
		if token.text == "fwd_ctx" && !allowed[i] {
			return fail("context parameter used outside an exact child call")
		}
		if token.text == "__cudair_ctx" {
			if i+3 >= len(tokens) || tokens[i+1].text != "." || tokens[i+3].text != "(" ||
				(tokens[i+2].text != "LoadByte" && tokens[i+2].text != "StoreByte" && tokens[i+2].text != "LoadFloat" && tokens[i+2].text != "StoreFloat") {
				return fail("context accessor used in an unsupported form")
			}
			if backwardSource && strings.HasPrefix(tokens[i+2].text, "Store") {
				return fail("backward context is read-only")
			}
			_, args, ok := callArguments(tokens, i+3)
			if !ok || len(args) != 2 {
				return fail("invalid context accessor arguments")
			}
			if strings.HasPrefix(tokens[i+2].text, "Load") {
				dest := tokens[args[1][0]:args[1][1]]
				if len(dest) != 2 || dest[0].text != "&" {
					return fail("context load destination must be an address of a scalar body local")
				}
				typeName := "uint8_t"
				if tokens[i+2].text == "LoadFloat" {
					typeName = "float"
				}
				declared := false
				for j := 0; j+1 < i; j++ {
					if tokens[j].text == typeName && tokens[j+1].text == dest[1].text {
						declared = true
					}
				}
				if !declared {
					return fail("context load destination is not a matching scalar body local")
				}
			}
			if value := integerConstant(tokens[args[0][0]:args[0][1]]); value != nil {
				offset, exact := constant.Uint64Val(value)
				width := uint64(1)
				if strings.HasSuffix(tokens[i+2].text, "Float") {
					width = 4
				}
				if !exact || uint64(ctxBytes) < width || offset > uint64(ctxBytes)-width || width == 4 && offset%4 != 0 {
					return fail("constant context offset is outside the declared region")
				}
			}
		}
	}
	return validateBodyWrites(node, tokens, children, backwardSource, fail)
}

// integerConstant evaluates the shared Go/C integer-expression subset. A
// symbol or unsupported C expression is dynamic and retains the runtime check.
func integerConstant(tokens []cudaToken) constant.Value {
	parts := make([]string, len(tokens))
	for i, t := range tokens {
		parts[i] = t.text
		if t.text[0] >= '0' && t.text[0] <= '9' {
			parts[i] = strings.TrimRight(t.text, "uUlL")
		}
	}
	value, err := types.Eval(token.NewFileSet(), nil, token.NoPos, strings.Join(parts, " "))
	if err != nil || value.Value == nil || value.Value.Kind() != constant.Int {
		return nil
	}
	return value.Value
}

func validateBodyWrites(node Emitter, tokens []cudaToken, children []ChildSymbol, backward bool, fail func(string) error) error {
	if err := validateDeclarationForms(tokens, fail); err != nil {
		return err
	}
	locals := make(map[string]bool)
	for i := 0; i+1 < len(tokens); i++ {
		if tokens[i].text == "float" || tokens[i].text == "int" || tokens[i].text == "bool" || tokens[i].text == "auto" || tokens[i].text == "uint8_t" {
			j := i + 1
			if tokens[j].text == "*" || tokens[j].text == "&" || tokens[j].text == "&&" {
				return fail("pointer or reference local is not allowed")
			}
			if tokens[j].text != "const" && isCUDAIdentStart(tokens[j].text[0]) {
				locals[tokens[j].text] = true
			}
		}
		if tokens[i].text == "*" && i+2 < len(tokens) && isCUDAIdentStart(tokens[i+1].text[0]) && (tokens[i+2].text == "=" || tokens[i+2].text == ";") {
			// A multiplication at the end of an initializer is not a pointer
			// declarator. Only a complete declaration prefix (or a leading
			// dereference store) gives this star pointer meaning.
			start := i - 1
			for start >= 0 && tokens[start].text != ";" && tokens[start].text != "{" && tokens[start].text != "}" {
				start--
			}
			start++
			pointer := start == i
			if start < i {
				pointer = true
				for j := start; j < i; j++ {
					if !isCUDAIdentStart(tokens[j].text[0]) || locals[tokens[j].text] || tokens[j].text == "return" {
						pointer = false
					}
				}
			}
			if !pointer {
				continue
			}
			if backward && tokens[i+2].text == "=" && (tokens[i+1].text == "dL_dx" || tokens[i+1].text == "dL_dparams") {
				return fail("direct gradient-pointer assignment is not supported: " + tokens[i+1].text)
			}
			return fail("pointer local is not allowed")
		}
		if tokens[i].text == ">" && i+2 < len(tokens) && isCUDAIdentStart(tokens[i+1].text[0]) && tokens[i+2].text == ";" {
			locals[tokens[i+1].text] = true
		}
	}
	childNames := make(map[string]bool)
	for _, child := range children {
		childNames[child.Name] = true
	}
	for name := range childNames {
		if locals[name] {
			return fail("local shadows an emitted child")
		}
	}
	keywords := map[string]bool{"if": true, "for": true, "while": true, "switch": true, "sizeof": true, "return": true, "static_cast": true, "reinterpret_cast": true, "const_cast": true, "lane_id": true}
	for i := 0; i+1 < len(tokens); i++ {
		if tokens[i+1].text == "(" && isCUDAIdentStart(tokens[i].text[0]) {
			name := tokens[i].text
			mathCall := name == "expf" || name == "erff" || name == "sqrtf" || name == "rsqrtf"
			if mathCall {
				if locals[name] || i > 0 && (tokens[i-1].text == "." || tokens[i-1].text == "::" || tokens[i-1].text == "->") {
					return fail("math intrinsic must be an unshadowed value call")
				}
				_, args, ok := callArguments(tokens, i+1)
				if !ok || len(args) != 1 || args[0][0] == args[0][1] {
					return fail("math intrinsic requires one value argument")
				}
				for j := args[0][0]; j < args[0][1]; j++ {
					text := tokens[j].text
					if text == "&" || text == "fwd_ctx" || text == "__cudair_ctx" || text == "nullptr" || text == "reinterpret_cast" || text == "const_cast" || text == "dynamic_cast" {
						return fail("math intrinsic cannot receive a pointer or context")
					}
					if text == "input" || text == "params" || strings.HasPrefix(text, "dL_d") {
						if j+1 >= args[0][1] || tokens[j+1].text != "[" {
							return fail("math intrinsic requires parameter values, not buffers")
						}
					}
				}
			}
			contextCall := i >= 2 && tokens[i-1].text == "." && tokens[i-2].text == "__cudair_ctx" && (name == "LoadByte" || name == "StoreByte" || name == "LoadFloat" || name == "StoreFloat")
			if !keywords[name] && !childNames[name] && name != "__expf" && !mathCall && !contextCall {
				return fail("call is outside the generated write allowlist")
			}
		}
	}
	for i, token := range tokens {
		switch token.text {
		case "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=":
			start := i - 1
			for start >= 0 && tokens[start].text != ";" && tokens[start].text != "{" && tokens[start].text != "}" {
				start--
			}
			start++
			if token.text == "=" {
				isAuto := false
				for j := start; j < i; j++ {
					if tokens[j].text == "auto" {
						isAuto = true
					}
				}
				rhsCall := i+1 < len(tokens) && childNames[tokens[i+1].text]
				if isAuto && !rhsCall {
					for j := i + 1; j < len(tokens) && tokens[j].text != ";"; j++ {
						if tokens[j].text == "params" || tokens[j].text == "input" || tokens[j].text == "fwd_ctx" || strings.HasPrefix(tokens[j].text, "dL_d") {
							return fail("auto local aliases a pointer parameter")
						}
					}
				}
			}
			root := ""
			bracket := false
			for j := start; j < i; j++ {
				if tokens[j].text == "params" || tokens[j].text == "input" || tokens[j].text == "fwd_ctx" || tokens[j].text == "dL_dy" {
					return fail("write target is a parameter or context pointer")
				}
				if tokens[j].text == "[" && !bracket {
					bracket = true
				}
				if isCUDAIdentStart(tokens[j].text[0]) && tokens[j].text != "const" && tokens[j].text != "float" && tokens[j].text != "int" && tokens[j].text != "auto" && tokens[j].text != "for" {
					if !bracket || root == "" {
						root = tokens[j].text
					}
				}
			}
			if root != "" && !locals[root] {
				if backward && (root == "dL_dx" || root == "dL_dparams") {
					indirect := bracket
					for j := start; j < i; j++ {
						if tokens[j].text == "*" {
							indirect = true
						}
					}
					if indirect {
						continue
					}
				}
				for j := 1; j+1 < len(tokens); j++ {
					if tokens[j].text == root && isCUDAIdentStart(tokens[j-1].text[0]) && (tokens[j+1].text == ";" || tokens[j+1].text == "=") {
						return fail("unsupported declaration " + tokens[j-1].text + " " + root)
					}
				}
				return fail("write target " + root + " is not a body local")
			}
		case "++", "--":
			j := i - 1
			if j < 0 || !isCUDAIdentStart(tokens[j].text[0]) {
				j = i + 1
			}
			if j < 0 || j >= len(tokens) || !locals[tokens[j].text] {
				return fail("increment target is not a body local")
			}
		}
	}
	return nil
}

// validateDeclarationForms recognizes complete declaration prefixes before
// reporting unsupported initializers or multiple declarators. It does not
// infer write targets from punctuation alone.
func validateDeclarationForms(tokens []cudaToken, fail func(string) error) error {
	for i := 0; i < len(tokens); i++ {
		if i > 0 && tokens[i-1].text != ";" && tokens[i-1].text != "{" && tokens[i-1].text != "}" {
			continue
		}
		start := i
		if tokens[start].text == "const" {
			start++
		}
		if start+1 >= len(tokens) || !isCUDAIdentStart(tokens[start].text[0]) {
			continue
		}
		name := start + 1
		if tokens[name].text == "<" {
			depth := 1
			for name++; name < len(tokens) && depth > 0; name++ {
				switch tokens[name].text {
				case "<":
					depth++
				case ">":
					depth--
				case ">>":
					depth -= 2
				}
			}
			if depth == 0 && name+1 < len(tokens) && isCUDAIdentStart(tokens[name].text[0]) && tokens[name+1].text == "=" {
				return fail("template initializer is not supported: " + tokens[name].text)
			}
			continue
		}
		switch tokens[start].text {
		case "float", "int", "bool", "uint8_t", "auto":
		default:
			continue
		}
		if !isCUDAIdentStart(tokens[name].text[0]) {
			continue
		}
		depth := 0
		for j := name + 1; j < len(tokens); j++ {
			switch tokens[j].text {
			case "(", "[", "{":
				depth++
			case ")", "]", "}":
				depth--
			case ";":
				if depth == 0 {
					j = len(tokens)
				}
			case ",":
				if depth == 0 && j+1 < len(tokens) && isCUDAIdentStart(tokens[j+1].text[0]) {
					return fail("comma-separated declaration is not supported: " + tokens[j+1].text)
				}
			}
		}
	}
	return nil
}

func balancedTokens(tokens []cudaToken) bool {
	var stack []string
	for _, token := range tokens {
		switch token.text {
		case "(", "[", "{":
			stack = append(stack, token.text)
		case ")", "]", "}":
			if len(stack) == 0 {
				return false
			}
			want := map[string]string{")": "(", "]": "[", "}": "{"}[token.text]
			if stack[len(stack)-1] != want {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}

func emittedFunctionName(source string) (string, bool) {
	for _, marker := range []string{"__device__ auto ", "__device__ void "} {
		if !strings.HasPrefix(source, marker) {
			continue
		}
		rest := source[len(marker):]
		i := strings.IndexByte(rest, '(')
		if i <= 0 || strings.ContainsAny(rest[:i], " \t\n") {
			return "", false
		}
		return rest[:i], true
	}
	return "", false
}

type cudaToken struct {
	text  string
	start int
	end   int
}

func tokenizeCUDA(source string) ([]cudaToken, error) {
	var out []cudaToken
	for i := 0; i < len(source); {
		c := source[i]
		if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
			i++
			continue
		}
		if i+1 < len(source) && source[i:i+2] == "//" {
			for i < len(source) && source[i] != '\n' {
				i++
			}
			continue
		}
		if i+1 < len(source) && source[i:i+2] == "/*" {
			j := strings.Index(source[i+2:], "*/")
			if j < 0 {
				return nil, fmt.Errorf("unterminated comment")
			}
			i += j + 4
			continue
		}
		start := i
		if isCUDAIdentStart(c) {
			i++
			for i < len(source) && isCUDAIdent(source[i]) {
				i++
			}
		} else if c >= '0' && c <= '9' {
			i++
			for i < len(source) && (isCUDAIdent(source[i]) || source[i] == '.') {
				i++
			}
		} else if c == '"' || c == '\'' {
			quote := c
			i++
			for i < len(source) {
				if source[i] == '\\' {
					i += 2
					continue
				}
				if source[i] == quote {
					i++
					break
				}
				i++
			}
			if i > len(source) || source[i-1] != quote {
				return nil, fmt.Errorf("unterminated literal")
			}
		} else {
			matched := ""
			for _, op := range []string{"<<=", ">>=", "++", "--", "->", "::", "+=", "-=", "*=", "/=", "%=", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "&=", "|=", "^="} {
				if strings.HasPrefix(source[i:], op) {
					matched = op
					break
				}
			}
			if matched != "" {
				i += len(matched)
			} else if strings.ContainsRune("(){}[];,.?:+-*/%&|^!~=<>", rune(c)) {
				i++
			} else {
				return nil, fmt.Errorf("unsupported token")
			}
		}
		out = append(out, cudaToken{text: source[start:i], start: start, end: i})
	}
	return out, nil
}

func callArguments(tokens []cudaToken, open int) (int, [][2]int, bool) {
	depth := 0
	start := open + 1
	var args [][2]int
	for i := open; i < len(tokens); i++ {
		switch tokens[i].text {
		case "(", "[", "{":
			depth++
		case ")":
			depth--
			if depth == 0 {
				if start < i {
					args = append(args, [2]int{start, i})
				}
				return i, args, true
			}
		case "]", "}":
			depth--
		case ",":
			if depth == 1 {
				args = append(args, [2]int{start, i})
				start = i + 1
			}
		}
	}
	return 0, nil, false
}

func tokenText(source string, tokens []cudaToken) string {
	if len(tokens) == 0 {
		return ""
	}
	return strings.TrimSpace(source[tokens[0].start:tokens[len(tokens)-1].end])
}

func isCUDAIdentStart(c byte) bool { return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }
func isCUDAIdent(c byte) bool      { return isCUDAIdentStart(c) || c >= '0' && c <= '9' }
