package marker

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

const prefixV3 = "cudaprof:v3 "

// ComponentContract declares component inputs and arithmetic. Hashes bind the
// source bank and declared root set; they do not prove roots were submitted,
// outputs were correct, or GPU work completed. Shape and signed element
// strides describe each root input. MathForm and ScalarPolicy are explicit
// application declarations, not reconstructed arithmetic. The zero value is
// not a valid contract for Encode.
type ComponentContract struct {
	Name            string `json:"name"`
	BankSHA256      string `json:"bank_sha256"`
	RootSetSHA256   string `json:"root_set_sha256"`
	Shape           []int  `json:"shape"`
	Strides         []int  `json:"strides"`
	DType           string `json:"dtype"`
	MathForm        string `json:"math_form"`
	ScalarPolicy    string `json:"scalar_policy"`
	ExecutionPolicy string `json:"execution_policy"`
	RootPolicy      string `json:"root_policy"`
}

func validateComponent(m Metadata) error {
	if m.Token != nil || m.Layer != nil || m.CacheLength != nil || m.Lookahead != nil || m.Materialization != nil || m.DType != nil {
		return errors.New("component metadata cannot contain model-token fields")
	}
	if strings.TrimSpace(m.RequestID) == "" || (m.Role != "warmup" && m.Role != "measured" && m.Role != "control") {
		return errors.New("component requires request id and valid role")
	}
	if m.Component == nil || m.Invocation == nil || m.Batch == nil || m.RootCount == nil {
		return errors.New("component requires contract, invocation, batch and root count")
	}
	if *m.Invocation < 0 || *m.Batch < 0 || *m.RootCount <= 0 || (m.Root != nil && (*m.Root < 0 || *m.Root >= *m.RootCount)) {
		return errors.New("component ordinals or root count are invalid")
	}
	c := m.Component
	if strings.TrimSpace(c.Name) == "" || strings.TrimSpace(c.DType) == "" || strings.TrimSpace(c.MathForm) == "" || strings.TrimSpace(c.ScalarPolicy) == "" || strings.TrimSpace(m.Operation) == "" {
		return errors.New("component name, dtype, math, scalar policy and operation must be nonempty")
	}
	for _, hash := range []string{c.BankSHA256, c.RootSetSHA256} {
		if len(hash) != 64 || strings.Trim(hash, "0123456789abcdef") != "" {
			return errors.New("component hashes must be lowercase sha256")
		}
	}
	if len(c.Shape) == 0 || len(c.Shape) != len(c.Strides) {
		return errors.New("component shape and strides must have equal nonzero rank")
	}
	for _, n := range c.Shape {
		if n <= 0 {
			return errors.New("component dimensions must be positive")
		}
	}
	if c.RootPolicy != "shared_aliases" && c.RootPolicy != "shared_input_distinct_roots" && c.RootPolicy != "distinct_banks" {
		return errors.New("component root policy is unsupported")
	}
	if c.ExecutionPolicy != "synchronized_eval" && c.ExecutionPolicy != "asynchronous_submission" {
		return errors.New("component execution policy is unsupported")
	}
	if m.Phase != "construct" && m.Phase != "submit" && m.Phase != "evaluate" && m.Phase != "cleanup" {
		return errors.New("component phase is unsupported")
	}
	if m.Event != "" && m.Event != "batch_start" && m.Event != "batch_end" {
		return errors.New("component event is unsupported")
	}
	if m.Boundary != "" && m.Boundary != "submission" && m.Boundary != "completion" {
		return errors.New("component boundary is unsupported")
	}
	if m.ExecutionUnit != "component_batch" {
		return errors.New("component execution unit must be component_batch")
	}
	if m.Executions != nil && *m.Executions < 0 {
		return errors.New("component executions must be nonnegative")
	}
	if m.Event == "batch_start" && (m.Boundary != "submission" || m.Executions == nil || m.Root != nil) {
		return errors.New("component batch start requires submission, counter and batch scope")
	}
	if m.Event == "batch_end" && (m.Boundary != "completion" || m.Executions == nil || m.Root != nil) {
		return errors.New("component batch end requires completion, counter and batch scope")
	}
	return nil
}

// The standard decoder rejects unknown fields but accepts duplicate keys and
// case aliases. Keep the nested contract as strict as the top-level schema.
func validateComponentJSON(raw []byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	t, err := dec.Token()
	if err != nil || t != json.Delim('{') {
		return errors.New("component must be an object")
	}
	seen := map[string]bool{}
	for dec.More() {
		t, err := dec.Token()
		if err != nil {
			return err
		}
		key, ok := t.(string)
		if !ok || seen[key] {
			return errors.New("component has duplicate or invalid field")
		}
		switch key {
		case "name", "bank_sha256", "root_set_sha256", "shape", "strides", "dtype", "math_form", "scalar_policy", "execution_policy", "root_policy":
		default:
			return fmt.Errorf("component has unknown field %q", key)
		}
		seen[key] = true
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return err
		}
		if key == "shape" || key == "strides" {
			var elements []json.RawMessage
			if err := json.Unmarshal(value, &elements); err != nil {
				return err
			}
			for _, element := range elements {
				if bytes.Equal(bytes.TrimSpace(element), []byte("null")) {
					return fmt.Errorf("component has null %s element", key)
				}
			}
		}
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return fmt.Errorf("component has null field %q", key)
		}
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		return errors.New("component has trailing data")
	}
	return nil
}
