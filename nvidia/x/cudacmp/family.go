package cudacmp

import (
	"context"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudalowering"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

// Architectural families defined for differential profiling.
const (
	FamilyExperts   = "experts"
	FamilyDenseQ8   = "dense_q8"
	FamilyF16       = "F16"
	FamilyAttention = "attention"
	FamilyNormPrep  = "norm_prep"
	FamilyModelPrep = "model_prep"
	FamilyResidual  = "residual"
)

// CanonicalFamilies lists the canonical families in standard display order.
var CanonicalFamilies = []string{
	FamilyExperts,
	FamilyDenseQ8,
	FamilyF16,
	FamilyAttention,
	FamilyNormPrep,
	FamilyModelPrep,
	FamilyResidual,
}

// FamilyClassifier normalizes kernel symbols and assigns them to architectural families.
type FamilyClassifier struct {
	Demangler *cudasass.Demangler
}

// NewFamilyClassifier creates a FamilyClassifier. If d is nil, pure-Go symbol
// normalization is used.
func NewFamilyClassifier(d *cudasass.Demangler) *FamilyClassifier {
	return &FamilyClassifier{Demangler: d}
}

// DefaultFamilyClassifier is a classifier using pure-Go normalization.
var DefaultFamilyClassifier = NewFamilyClassifier(nil)

// Demangle resolves mangled symbols using the configured demangler, or returns
// symbols as-is if no demangler is configured.
func (fc *FamilyClassifier) Demangle(ctx context.Context, symbols []string) ([]string, error) {
	if fc != nil && fc.Demangler != nil {
		out, err := fc.Demangler.Demangle(ctx, symbols)
		if err != nil {
			return nil, err
		}
		lines := strings.Split(strings.TrimRight(string(out.Stdout), "\r\n"), "\n")
		if len(lines) == len(symbols) {
			return lines, nil
		}
	}
	return symbols, nil
}

// Normalize reduces a mangled symbol or demangled C++ function signature down to its
// canonical base function identifier.
func (fc *FamilyClassifier) Normalize(symbol string) string {
	s := strings.TrimSpace(symbol)
	if s == "" {
		return ""
	}
	// If it starts with Itanium mangling, cudalowering.MatchKey decodes the qualified name.
	if strings.HasPrefix(s, "_Z") {
		k := cudalowering.MatchKey(s)
		if k != "" {
			if idx := strings.LastIndex(k, "::"); idx >= 0 {
				return k[idx+2:]
			}
			return k
		}
	}

	// Clean demangled C++ signatures:
	s = strings.ReplaceAll(s, "(anonymous namespace)::", "")
	s = strings.ReplaceAll(s, "<unnamed>::", "")

	// Strip parameter list: trailing "(...)"
	if strings.HasSuffix(s, ")") {
		depth := 0
		for i := len(s) - 1; i >= 0; i-- {
			if s[i] == ')' {
				depth++
			} else if s[i] == '(' {
				depth--
				if depth == 0 {
					s = strings.TrimSpace(s[:i])
					break
				}
			}
		}
	}

	// Strip template arguments: trailing "<...>"
	s = stripTemplateArgs(s)

	// Strip return types or storage classes preceding the function identifier:
	// e.g. "void foo" -> "foo", "__global__ void bar" -> "bar"
	if fields := strings.Fields(s); len(fields) > 1 {
		s = fields[len(fields)-1]
	}

	// Strip namespace qualifiers: "a::b::c" -> "c"
	if idx := strings.LastIndex(s, "::"); idx >= 0 {
		s = s[idx+2:]
	}

	return strings.Trim(s, "*& \t\r\n")
}

// stripTemplateArgs strips balanced '<' ... '>' at the end of a string.
func stripTemplateArgs(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasSuffix(s, ">") {
		return s
	}
	depth := 0
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '>' {
			depth++
		} else if s[i] == '<' {
			depth--
			if depth == 0 {
				return strings.TrimSpace(s[:i])
			}
		}
	}
	return s
}

// Classify categorizes a kernel symbol into an architectural family.
func (fc *FamilyClassifier) Classify(symbol string) string {
	norm := fc.Normalize(symbol)
	low := strings.ToLower(norm)
	rawLow := strings.ToLower(symbol)

	// 1. Model preparation (offline weight repacking / quantization setup)
	if strings.Contains(low, "repack") || strings.Contains(low, "model_prep") || strings.Contains(low, "offline_prep") {
		return FamilyModelPrep
	}

	// 2. Experts (MoE routed kernels: gateup, down, router, swiglu, iq2, q2_k)
	if strings.Contains(low, "gateup") ||
		strings.Contains(low, "swiglu") ||
		strings.Contains(low, "router") ||
		strings.Contains(low, "iq2_xxs") ||
		strings.Contains(low, "iq2") ||
		strings.Contains(low, "q2_k") ||
		strings.Contains(low, "moe") ||
		strings.Contains(low, "expert") ||
		strings.Contains(rawLow, "gateup") ||
		strings.Contains(rawLow, "swiglu") ||
		isDownMoEKernel(low) {
		return FamilyExperts
	}

	// 3. Attention (RoPE, attention score tiling, KV cache, SDPA)
	if strings.Contains(low, "attention") ||
		strings.Contains(low, "scores_tile") ||
		strings.Contains(low, "rope") ||
		strings.Contains(low, "kv_cache") ||
		strings.Contains(low, "sdpa") ||
		strings.Contains(low, "attn") ||
		strings.Contains(low, "softmax") {
		return FamilyAttention
	}

	// 4. Normalization and preparation (RMSNorm, dequantization, layout conversion)
	if strings.Contains(low, "rms_norm") ||
		strings.Contains(low, "rmsnorm") ||
		strings.Contains(low, "dequant") ||
		strings.Contains(low, "f32_to_f16") ||
		strings.Contains(low, "f16_to_f32") ||
		strings.Contains(low, "quantize") ||
		strings.Contains(low, "norm_prep") ||
		strings.Contains(low, "layernorm") ||
		strings.Contains(low, "layer_norm") {
		return FamilyNormPrep
	}

	// 5. Dense Q8 (attention dense projection, Q8 GEMMs/GEMVs)
	if strings.Contains(low, "dense_q8") ||
		strings.Contains(low, "mul_mat_q") ||
		strings.Contains(low, "mmq") ||
		strings.Contains(low, "q8_0") ||
		strings.Contains(low, "q8_1") ||
		strings.Contains(low, "gemm_q8") ||
		strings.Contains(low, "q8") {
		return FamilyDenseQ8
	}

	// 6. F16 (half precision / bfloat16 projections and GEMMs)
	if strings.Contains(low, "f16") ||
		strings.Contains(low, "fp16") ||
		strings.Contains(low, "half") ||
		strings.Contains(low, "bfloat16") ||
		strings.Contains(low, "bf16") {
		return FamilyF16
	}

	// 7. Residual (unclassified, elementwise addition, copy, etc.)
	return FamilyResidual
}

func isDownMoEKernel(s string) bool {
	return strings.HasPrefix(s, "down_") ||
		strings.HasSuffix(s, "_down") ||
		strings.Contains(s, "_down_") ||
		s == "down"
}
