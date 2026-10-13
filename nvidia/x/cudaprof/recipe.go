package cudaprof

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// RecipeSchemaVersion defines the schema version for cudainspect reports.
const RecipeSchemaVersion = 1

// Feature observation states according to cudainspect schema v1.
const (
	StateObserved    = "observed"
	StateInferred    = "inferred"
	StateUnknown     = "unknown"
	StateUnsupported = "unsupported"
	StateError       = "error"
)

// RecipeReport represents a CUBIN inspection report conforming to the cudainspect JSON schema v1.
type RecipeReport struct {
	SchemaVersion int                 `json:"schema_version"`
	Artifact      RecipeArtifact      `json:"artifact"`
	Selection     RecipeSelection     `json:"selection"`
	Target        RecipeTarget        `json:"target"`
	Request       RecipeRequest       `json:"request"`
	Result        string              `json:"result"` // "complete", "partial", or "failed"
	Features      []RecipeFeature     `json:"features"`
	Instructions  []RecipeInstruction `json:"instructions,omitempty"`
	Diagnostics   []RecipeDiagnostic  `json:"diagnostics,omitempty"`
	Receipts      []RecipeReceipt     `json:"receipts,omitempty"`
}

type RecipeArtifact struct {
	SHA256     string `json:"sha256"`
	SizeBytes  int64  `json:"size_bytes"`
	ParentHash string `json:"parent_hash,omitempty"`
	ImageName  string `json:"image_name,omitempty"`
}

type RecipeSelection struct {
	Symbol   string `json:"symbol"`
	Function string `json:"function"`
	Scope    string `json:"scope,omitempty"`
}

type RecipeTarget struct {
	RawArchitecture string `json:"raw_architecture"`
	ComputeCap      string `json:"compute_cap"`
	FeatureSuffix   string `json:"feature_suffix,omitempty"`
}

type RecipeRequest struct {
	Required []string          `json:"required"`
	Optional []string          `json:"optional,omitempty"`
	Limits   map[string]string `json:"limits,omitempty"`
}

type RecipeFeature struct {
	Name       string   `json:"name"`
	State      string   `json:"state"` // observed, inferred, unknown, unsupported, error
	Value      *float64 `json:"value"` // nullable; explicit null when unknown/unsupported/error
	Unit       string   `json:"unit,omitempty"`
	Evidence   []string `json:"evidence,omitempty"`
	Derivation string   `json:"derivation,omitempty"`
}

type RecipeInstruction struct {
	PC         uint64 `json:"pc"`
	Predicate  string `json:"predicate,omitempty"`
	Opcode     string `json:"opcode"`
	Operands   string `json:"operands,omitempty"`
	WidthBits  int    `json:"width_bits,omitempty"` // 128 for LDG.E.128, 32 for scalar LDG.E
	IsVector   bool   `json:"is_vector"`
	IsLocalMem bool   `json:"is_local_mem"`
	IsSpill    bool   `json:"is_spill"`
}

type RecipeDiagnostic struct {
	Code     string   `json:"code"`
	Feature  string   `json:"feature"`
	Message  string   `json:"message"`
	Evidence []string `json:"evidence,omitempty"`
}

type RecipeReceipt struct {
	Tool       string `json:"tool"`
	Version    string `json:"version"`
	ExitStatus int    `json:"exit_status"`
	DurationMs int64  `json:"duration_ms"`
}

// WriteJSON writes the report to w formatted with indentation.
func (r *RecipeReport) WriteJSON(w io.Writer) error {
	if r == nil {
		return errors.New("cudaprof: nil recipe report")
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// ReadRecipeReport decodes a JSON recipe report from r.
func ReadRecipeReport(r io.Reader) (*RecipeReport, error) {
	var rep RecipeReport
	dec := json.NewDecoder(r)
	if err := dec.Decode(&rep); err != nil {
		return nil, fmt.Errorf("cudaprof: decode recipe report: %w", err)
	}
	return &rep, nil
}

// Validate verifies schema integrity, ensuring unknown values are null with diagnostic reasons,
// and allocated registers and peak live GPR remain distinct.
func (r *RecipeReport) Validate() error {
	if r.SchemaVersion != RecipeSchemaVersion {
		return fmt.Errorf("cudaprof: unsupported recipe schema version %d (want %d)", r.SchemaVersion, RecipeSchemaVersion)
	}
	if r.Artifact.SHA256 == "" {
		return errors.New("cudaprof: missing artifact sha256")
	}
	if r.Selection.Symbol == "" {
		return errors.New("cudaprof: missing selection symbol")
	}

	diagFeatures := make(map[string]bool)
	for _, d := range r.Diagnostics {
		if d.Feature != "" {
			diagFeatures[d.Feature] = true
		}
	}

	for _, f := range r.Features {
		switch f.State {
		case StateUnknown, StateUnsupported, StateError:
			if f.Value != nil {
				return fmt.Errorf("cudaprof: feature %q has state %q but non-null value %v (unknowns must be null)", f.Name, f.State, *f.Value)
			}
			if !diagFeatures[f.Name] {
				return fmt.Errorf("cudaprof: unknown feature %q requires diagnostic reason in diagnostics", f.Name)
			}
		case StateObserved, StateInferred:
			if f.Value == nil {
				return fmt.Errorf("cudaprof: feature %q has state %q but null value", f.Name, f.State)
			}
			if len(f.Evidence) == 0 {
				return fmt.Errorf("cudaprof: feature %q with state %q requires evidence locator", f.Name, f.State)
			}
		default:
			return fmt.Errorf("cudaprof: unrecognized feature state %q for feature %q", f.State, f.Name)
		}
	}

	return nil
}

// Float64Ptr returns a pointer to the given float64 value.
func Float64Ptr(v float64) *float64 {
	return &v
}
