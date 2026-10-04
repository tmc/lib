package cudalayer_test

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalayer"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// PinnedFusedMLP64SourceHash pins the canonical source output of the 3-layer width-64 MLP.
// Computed via sha256.Sum256([]byte(unit.Source)).
const PinnedFusedMLP64SourceHash = "78250957fac0ab1a7fe9ceeef93bccbbf288908e6cebd2d7e23bffff6c0ad1b9"

func TestFusedMLP64_EmitsSingleDeviceFunction(t *testing.T) {
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatalf("NewFusedMLP: %v", err)
	}

	unit, err := cudair.EmitTree(mlp, "fused_mlp64_eval")
	if err != nil {
		t.Fatalf("EmitTree: %v", err)
	}

	// Every emitted symbol is defined exactly once and the parent calls the
	// selected child symbols once in order, using consecutive parameter regions.
	if len(unit.Symbols) != 4 {
		t.Fatalf("expected 4 symbols (3 layers + root), got %d: %v", len(unit.Symbols), unit.Symbols)
	}

	// Verify the top-level device function signature
	parentSig := "__device__ auto fused_mlp64_eval(const tvec<float, 64>& input, const float* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> tvec<float, 64>"
	if !strings.Contains(unit.Source, parentSig) {
		t.Fatalf("expected parent signature:\n%s\ngot:\n%s", parentSig, unit.Source)
	}

	for _, symbol := range unit.Symbols {
		if n := strings.Count(unit.Source, "__device__ auto "+symbol+"("); n != 1 {
			t.Fatalf("symbol %q has %d definitions", symbol, n)
		}
	}
	parentStart := strings.Index(unit.Source, parentSig)
	if parentStart < 0 {
		t.Fatalf("parent body missing:\n%s", unit.Source)
	}
	parentEnd := strings.Index(unit.Source[parentStart:], "\n}") + parentStart
	if parentEnd < parentStart {
		t.Fatalf("parent body missing:\n%s", unit.Source)
	}
	parentBody := unit.Source[parentStart:parentEnd]
	for i, input := range []string{"input", "h0", "h1"} {
		ctx := "fwd_ctx"
		if i > 0 {
			ctx = fmt.Sprintf("(fwd_ctx ? fwd_ctx + WARP_SIZE * %d : nullptr)", i*512)
		}
		call := fmt.Sprintf("%s(%s, params + %d, %s)", unit.Symbols[i], input, i*64*64, ctx)
		if strings.Count(parentBody, call) != 1 {
			t.Fatalf("parent must call %s once: %s", call, parentBody)
		}
		if i > 0 && strings.Index(parentBody, call) < strings.Index(parentBody, unit.Symbols[i-1]+"(") {
			t.Fatalf("child calls are out of order: %s", parentBody)
		}
	}

	// Splice into outer kernel template
	subs := map[string]string{
		"FORWARD_DEVICE_FUNCTION": unit.Source,
		"KERNEL_NAME":             "mlp64_kernel",
		"FUNC_NAME":               "fused_mlp64_eval",
		"IN_WIDTH":                "64",
		"OUT_WIDTH":               "64",
		"LOSS_DEVICE_FUNCTION":    "",
	}

	spliced, err := nvrtcx.SpliceTemplate(cudalayer.OuterKernelTemplate, subs)
	if err != nil {
		t.Fatalf("SpliceTemplate failed: %v", err)
	}

	if !strings.Contains(spliced, "__global__ void mlp64_kernel(") {
		t.Fatalf("spliced code missing global kernel: %s", spliced)
	}
}

func TestOuterKernelTemplateDefinesEmitterHelpers(t *testing.T) {
	if strings.Contains(cudalayer.OuterKernelTemplate, "#include") {
		t.Fatal("NVRTC template depends on external headers")
	}
	if !strings.Contains(cudalayer.OuterKernelTemplate, "using uint8_t = unsigned char;") {
		t.Fatal("NVRTC template does not define its context byte type")
	}
	for _, helper := range []string{
		"struct tvec {",
		"static constexpr int WARP_SIZE = 32;",
		"#define lane_id()",
	} {
		if !strings.Contains(cudalayer.OuterKernelTemplate, helper) {
			t.Errorf("outer kernel template is missing emitter helper %q", helper)
		}
	}
}

func TestFusedMLP64_NoIntermediateGlobalAllocation(t *testing.T) {
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatalf("NewFusedMLP: %v", err)
	}

	unit, err := cudair.EmitTree(mlp, "fused_mlp64_eval")
	if err != nil {
		t.Fatalf("EmitTree: %v", err)
	}

	parentStart := strings.Index(unit.Source, "__device__ auto fused_mlp64_eval(")
	if parentStart < 0 {
		t.Fatal("parent function body missing")
	}
	parentEnd := strings.Index(unit.Source[parentStart:], "\n}") + parentStart
	if parentEnd < parentStart {
		t.Fatal("parent function body missing")
	}
	parentLines := nonemptyLines(unit.Source[parentStart:parentEnd])
	if len(parentLines) != 4 || !strings.HasPrefix(parentLines[1], "auto h0 = ") || !strings.HasPrefix(parentLines[2], "auto h1 = ") || !strings.HasPrefix(parentLines[3], "return ") {
		t.Fatalf("parent must contain only two local activations and a return: %v", parentLines)
	}
	for _, symbol := range unit.Symbols[:3] {
		start := strings.Index(unit.Source, "__device__ auto "+symbol+"(")
		if start < 0 {
			t.Fatalf("child %q body missing", symbol)
		}
		end := strings.Index(unit.Source[start:], "\n}") + start
		if end < start {
			t.Fatalf("child %q body missing", symbol)
		}
		body := unit.Source[start:end]
		for _, forbidden := range []string{"out_data", "in_data", "cudaMalloc", "cudamem"} {
			if strings.Contains(body, forbidden) {
				t.Fatalf("child %q has global staging %q", symbol, forbidden)
			}
		}
		if !strings.Contains(body, "out[out_idx] = val;") {
			t.Fatalf("child %q does not write only its local output vector:\n%s", symbol, body)
		}
		// The generated bounded accessor is independently validated by cudair.
		// Inspect the supplied body after the fixed accessor declaration.
		accessor := strings.Index(body, "__cudair_context __cudair_ctx{")
		if accessor < 0 {
			t.Fatal("missing bounded context accessor")
		}
		body = body[accessor+strings.IndexByte(body[accessor:], '\n')+1:]
		if hasNonAllowlistedWrite(body, "out", "val", "out_idx", "in_idx", "saved_input", "saved_value") {
			t.Fatalf("child %q has a write outside body locals:\n%s", symbol, body)
		}
	}
	subs := map[string]string{"FORWARD_DEVICE_FUNCTION": unit.Source, "LOSS_DEVICE_FUNCTION": "", "KERNEL_NAME": "mlp64_kernel", "FUNC_NAME": "fused_mlp64_eval", "IN_WIDTH": "64", "OUT_WIDTH": "64"}
	spliced, err := nvrtcx.SpliceTemplate(cudalayer.OuterKernelTemplate, subs)
	if err != nil {
		t.Fatal(err)
	}
	kernelStart := strings.Index(spliced, "extern \"C\" __global__")
	if kernelStart < 0 {
		t.Fatal("spliced outer kernel missing")
	}
	if hasNonAllowlistedWrite(spliced[kernelStart:], "out_data", "idx", "i", "in_val", "out_val") {
		t.Fatalf("outer kernel has a write outside locals and out_data:\n%s", spliced[kernelStart:])
	}
	if !strings.Contains(spliced[kernelStart:], "out_data[idx * 64 + i] =") {
		t.Fatal("outer kernel has no out_data store")
	}
}

func hasNonAllowlistedWrite(source string, allowed ...string) bool {
	if start := strings.IndexByte(source, '{'); start >= 0 {
		source = source[start+1:]
	}
	if end := strings.LastIndexByte(source, '}'); end >= 0 {
		source = source[:end]
	}
	tokens := cppTokens(source)
	allowedSet := make(map[string]bool, len(allowed))
	for _, name := range allowed {
		allowedSet[name] = true
	}
	aliases := make(map[string]bool)
	for i := 0; i+2 < len(tokens); i++ {
		if tokens[i] == "*" && isIdent(tokens[i+1]) && tokens[i+2] == "=" {
			for j := i + 3; j < len(tokens) && tokens[j] != ";"; j++ {
				if tokens[j] == "params" || tokens[j] == "in_data" || tokens[j] == "fwd_ctx" {
					aliases[tokens[i+1]] = true
				}
			}
		}
	}
	for i, tok := range tokens {
		if tok == "atomicAdd" || tok == "atomicExch" || tok == "memcpy" || tok == "memmove" {
			return true
		}
		if tok == "=" || tok == "+=" || tok == "-=" || tok == "*=" || tok == "/=" || tok == "%=" || tok == "++" || tok == "--" {
			if tok != "++" && tok != "--" {
				for k := i - 1; k >= 0 && tokens[k] != ";" && tokens[k] != "{" && tokens[k] != "}"; k-- {
					if tokens[k] == "params" || tokens[k] == "in_data" || tokens[k] == "fwd_ctx" {
						return true
					}
				}
			}
			j := i - 1
			if tok == "++" || tok == "--" {
				if j < 0 || !isIdent(tokens[j]) {
					j = i + 1
				}
			} else if j >= 0 && tokens[j] == "]" {
				depth := 1
				for j--; j >= 0; j-- {
					if tokens[j] == "]" {
						depth++
					}
					if tokens[j] == "[" {
						depth--
						if depth == 0 {
							j--
							break
						}
					}
				}
			}
			if j < 0 || !isIdent(tokens[j]) {
				continue
			}
			name := tokens[j]
			if aliases[name] || !allowedSet[name] {
				return true
			}
			for k := j - 1; k >= 0 && tokens[k] != ";" && tokens[k] != "{" && tokens[k] != "}"; k-- {
				if tokens[k] == "params" || tokens[k] == "in_data" || tokens[k] == "fwd_ctx" {
					return true
				}
				if tokens[k] == "[" {
					break
				}
			}
		}
	}
	return false
}

func cppTokens(source string) []string {
	var out []string
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
		if c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' {
			j := i + 1
			for j < len(source) && (source[j] == '_' || source[j] >= 'a' && source[j] <= 'z' || source[j] >= 'A' && source[j] <= 'Z' || source[j] >= '0' && source[j] <= '9') {
				j++
			}
			out = append(out, source[i:j])
			i = j
			continue
		}
		matched := ""
		for _, op := range []string{"<<=", ">>=", "++", "--", "+=", "-=", "*=", "/=", "%=", "->", "::", "==", "!=", "<=", ">=", "&&", "||"} {
			if strings.HasPrefix(source[i:], op) {
				matched = op
				break
			}
		}
		if matched != "" {
			out = append(out, matched)
			i += len(matched)
			continue
		}
		out = append(out, string(c))
		i++
	}
	return out
}

func isIdent(s string) bool {
	return s != "" && (s[0] == '_' || s[0] >= 'a' && s[0] <= 'z' || s[0] >= 'A' && s[0] <= 'Z')
}

func TestWriteAllowlistFindsStoresAndKeepsControls(t *testing.T) {
	for _, source := range []string{
		"params[i] += v;", "--in_data[i];", "const_cast<float*>(params)[idx] = v;", "*(in_data + idx) = v;", "float* w = const_cast<float*>(params); w[i] = v;",
		"atomicAdd(params + i, v);", "memcpy(in_data + i, src, n);", "params[i]\n = v;",
	} {
		if !hasNonAllowlistedWrite(source, "out_data", "local") {
			t.Errorf("accepted unallowlisted write %q", source)
		}
	}
	if hasNonAllowlistedWrite("out_data[idx] = v; local += 1; ++local;", "out_data", "local") {
		t.Fatal("rejected allowlisted output or local write")
	}
}

type badContextLayer struct {
	name    string
	in, out int
}

func (l *badContextLayer) Name() string                  { return l.name }
func (l *badContextLayer) InputWidth() int               { return l.in }
func (l *badContextLayer) OutputWidth() int              { return l.out }
func (l *badContextLayer) PaddedOutputWidth() int        { return l.out }
func (l *badContextLayer) FwdCtxBytes() (int, error)     { return 0, nil }
func (l *badContextLayer) FwdCtxAlignedPerElement() bool { return true }
func (l *badContextLayer) EmitForward(name string) (string, error) {
	return fmt.Sprintf("__device__ auto %s(const tvec<float, %d>& input, const float* __restrict__ params, uint8_t* __restrict__ c = nullptr) -> tvec<float, %d> { ((float*)c)[0] = 1; return input; }", name, l.in, l.out), nil
}

type width64BadMLP struct {
	*cudalayer.FusedMLP
	children []cudair.Emitter
}

func (m *width64BadMLP) Children() []cudair.Emitter { return m.children }

func TestWidth64MLPRejectsLayerContextWrite(t *testing.T) {
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatal(err)
	}
	children := mlp.Children()
	children[0] = &badContextLayer{name: "layer_0", in: 64, out: 64}
	root := &width64BadMLP{FusedMLP: mlp, children: children}
	_, err = cudair.EmitTree(root, "width64_eval")
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), "layer_0") {
		t.Fatalf("EmitTree error = %v, want ErrContextBoundsExceeded naming layer_0", err)
	}
}

func nonemptyLines(source string) []string {
	var lines []string
	for _, line := range strings.Split(source, "\n") {
		line = strings.TrimSpace(line)
		if line != "" && line != "{" {
			lines = append(lines, line)
		}
	}
	return lines
}

func TestFusedMLP64_SplicedKernelCompiles(t *testing.T) {
	major, minor := cudatest.RequireNVRTC(t)
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatal(err)
	}
	unit, err := cudair.EmitTree(mlp, "fused_mlp64_eval")
	if err != nil {
		t.Fatal(err)
	}
	backward, err := cudair.EmitBackwardTree(mlp, "fused_mlp64_bwd", 32, cudair.GradientModeOverwrite)
	if err != nil {
		t.Fatal(err)
	}
	for _, loss := range []string{"__device__ void optional_loss() {}", ""} {
		subs := map[string]string{"FORWARD_DEVICE_FUNCTION": unit.Source, "LOSS_DEVICE_FUNCTION": backward.Source + loss, "KERNEL_NAME": "mlp64_kernel", "FUNC_NAME": "fused_mlp64_eval", "IN_WIDTH": "64", "OUT_WIDTH": "64"}
		source, err := nvrtcx.SpliceTemplate(cudalayer.OuterKernelTemplate, subs)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := cudatest.CompileSource(source, major, minor); err != nil {
			t.Fatalf("spliced kernel (loss length %d) does not compile: %v", len(loss), err)
		}
	}
}

func TestFusedMLP64_BackwardTree(t *testing.T) {
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, child := range mlp.Children() {
		backward, ok := child.(cudair.BackwardEmitter)
		if !ok {
			t.Fatal("layer has no backward emitter")
		}
		unit, err := cudair.EmitBackwardTree(backward, "layer_bwd", 32, cudair.GradientModeOverwrite)
		if err != nil || !strings.Contains(unit.Source, "(*dL_dx)[in_idx] = grad") {
			t.Fatalf("layer source=%s error=%v", unit.Source, err)
		}
	}
	unit, err := cudair.EmitBackwardTree(mlp, "mlp_bwd", 32, cudair.GradientModeOverwrite)
	if err != nil {
		t.Fatal(err)
	}
	if len(unit.Symbols) != 4 {
		t.Fatalf("symbols=%v, want three layers and root", unit.Symbols)
	}
	start := strings.LastIndex(unit.Source, "__device__ void mlp_bwd(")
	if start < 0 {
		t.Fatal("missing parent backward function")
	}
	parent := unit.Source[start:]
	previous := -1
	for i := 2; i >= 0; i-- {
		position := strings.Index(parent, unit.Symbols[i]+"(")
		if position <= previous {
			t.Fatalf("backward calls not in reverse order: %s", parent)
		}
		previous = position
	}
}

func TestFusedMLP_BackwardParameterOffsetsAndGradientChain(t *testing.T) {
	mlp, err := cudalayer.NewFusedMLP(2, 3, 5, 3)
	if err != nil {
		t.Fatal(err)
	}
	forward, err := cudair.EmitTree(mlp, "forward_chain")
	if err != nil {
		t.Fatal(err)
	}
	backward, err := cudair.EmitBackwardTree(mlp, "backward_chain", 32, cudair.GradientModeOverwrite)
	if err != nil {
		t.Fatal(err)
	}
	for i, offset := range []int{0, 6, 15} {
		ctx := "fwd_ctx"
		if i > 0 {
			ctx = fmt.Sprintf("(fwd_ctx ? fwd_ctx + WARP_SIZE * %d : nullptr)", []int{0, 20, 44}[i])
		}
		input := "input"
		if i > 0 {
			input = fmt.Sprintf("h%d", i-1)
		}
		forwardCall := fmt.Sprintf("%s(%s, params + %d, %s)", forward.Symbols[i], input, offset, ctx)
		if !strings.Contains(forward.Source, forwardCall) {
			t.Fatalf("missing forward parameter layout %q:\n%s", forwardCall, forward.Source)
		}
		dy, dx := "dL_dy", "dL_dx"
		if i < 2 {
			dy = fmt.Sprintf("grad%d", i+1)
		}
		if i > 0 {
			dx = fmt.Sprintf("&grad%d", i)
			declaration := fmt.Sprintf("tvec<float, 3> grad%d;", i)
			if !strings.Contains(backward.Source, declaration) {
				t.Fatalf("missing intermediate gradient %q", declaration)
			}
		}
		backwardCall := fmt.Sprintf("%s(%s, params + %d, %s, (dL_dparams ? dL_dparams + %d : nullptr), %s)", backward.Symbols[i], dy, offset, ctx, offset, dx)
		if !strings.Contains(backward.Source, backwardCall) {
			t.Fatalf("missing backward parameter layout or gradient chain %q:\n%s", backwardCall, backward.Source)
		}
	}
}

func TestEmittedFunctions_OnlyFromEmitFromBody(t *testing.T) {
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatal(err)
	}
	unit, err := cudair.EmitTree(mlp, "fused_mlp64_eval")
	if err != nil {
		t.Fatal(err)
	}
	children := mlp.Children()
	for i, symbol := range unit.Symbols {
		var generated string
		if i < len(children) {
			generated, err = cudair.EmitFromBody(children[i], symbol, "")
		} else {
			generated, err = cudair.EmitFromBody(mlp, symbol, "")
		}
		if err != nil {
			t.Fatal(err)
		}
		want := strings.SplitN(generated, " {", 2)[0]
		definition := regexp.MustCompile(`(?m)^__device__ auto ` + regexp.QuoteMeta(symbol) + `\([^\n{]*`)
		match := strings.TrimSpace(definition.FindString(unit.Source))
		if match == "" || match != want {
			t.Fatalf("device definition for %q did not come from EmitFromBody:\n got %q\nwant %q", symbol, match, want)
		}
	}
	subs := map[string]string{"FORWARD_DEVICE_FUNCTION": unit.Source, "LOSS_DEVICE_FUNCTION": "", "KERNEL_NAME": "mlp64_kernel", "FUNC_NAME": "fused_mlp64_eval", "IN_WIDTH": "64", "OUT_WIDTH": "64"}
	spliced, err := nvrtcx.SpliceTemplate(cudalayer.OuterKernelTemplate, subs)
	if err != nil {
		t.Fatal(err)
	}
	definitions := regexp.MustCompile(`(?m)^__device__ (?:auto|void) [^\n]+`).FindAllString(spliced, -1)
	if len(definitions) != len(unit.Symbols) {
		t.Fatalf("spliced source has %d device definitions for %d emitted symbols: %v", len(definitions), len(unit.Symbols), definitions)
	}
}

func TestEmittedSourceHash_Pinned(t *testing.T) {
	mlp, err := cudalayer.NewFusedMLP(64, 64, 64, 3)
	if err != nil {
		t.Fatalf("NewFusedMLP: %v", err)
	}

	unit, err := cudair.EmitTree(mlp, "fused_mlp64_eval")
	if err != nil {
		t.Fatalf("EmitTree: %v", err)
	}

	h := sha256.Sum256([]byte(unit.Source))
	gotHash := hex.EncodeToString(h[:])

	if PinnedFusedMLP64SourceHash == "" {
		t.Logf("Generated source hash: %s", gotHash)
	} else if gotHash != PinnedFusedMLP64SourceHash {
		t.Fatalf("emitted source hash changed!\ngot:  %s\nwant: %s\nAny emission change must be reviewed and deliberately updated.",
			gotHash, PinnedFusedMLP64SourceHash)
	}
}
