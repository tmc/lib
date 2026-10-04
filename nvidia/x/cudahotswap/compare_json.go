package cudahotswap

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
)

type mismatchJSON struct {
	Index  int        `json:"index"`
	Got    jsonNumber `json:"got"`
	Want   jsonNumber `json:"want"`
	AbsErr jsonNumber `json:"abs_err"`
	RelErr jsonNumber `json:"rel_err"`
}

// MarshalJSON preserves nonfinite failure evidence using explicit strings.
// Finite values retain their ordinary JSON number representation.
func (m Mismatch) MarshalJSON() ([]byte, error) {
	return json.Marshal(mismatchJSON{m.Index, jsonNumber(m.Got), jsonNumber(m.Want), jsonNumber(m.AbsErr), jsonNumber(m.RelErr)})
}

// UnmarshalJSON accepts ordinary numbers and the three nonfinite strings
// emitted by MarshalJSON. An invalid value leaves m unchanged.
func (m *Mismatch) UnmarshalJSON(b []byte) error {
	if m == nil {
		return fmt.Errorf("decode mismatch: nil destination")
	}
	var v struct {
		Index  *int        `json:"index"`
		Got    *jsonNumber `json:"got"`
		Want   *jsonNumber `json:"want"`
		AbsErr *jsonNumber `json:"abs_err"`
		RelErr *jsonNumber `json:"rel_err"`
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return fmt.Errorf("decode mismatch: %w", err)
	}
	if v.Index == nil || v.Got == nil || v.Want == nil || v.AbsErr == nil || v.RelErr == nil {
		return fmt.Errorf("decode mismatch: missing comparison field")
	}
	*m = Mismatch{*v.Index, float64(*v.Got), float64(*v.Want), float64(*v.AbsErr), float64(*v.RelErr)}
	return nil
}

type jsonNumber float64

func (v jsonNumber) MarshalJSON() ([]byte, error) {
	switch f := float64(v); {
	case math.IsNaN(f):
		return []byte(`"NaN"`), nil
	case math.IsInf(f, 1):
		return []byte(`"+Inf"`), nil
	case math.IsInf(f, -1):
		return []byte(`"-Inf"`), nil
	default:
		return json.Marshal(f)
	}
}

func (v *jsonNumber) UnmarshalJSON(b []byte) error {
	b = bytes.TrimSpace(b)
	if string(b) == "null" {
		return fmt.Errorf("comparison number is null")
	}
	var f float64
	if len(b) > 0 && b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		switch s {
		case "NaN":
			f = math.NaN()
		case "+Inf":
			f = math.Inf(1)
		case "-Inf":
			f = math.Inf(-1)
		default:
			return fmt.Errorf("unknown comparison number %q", s)
		}
	} else if err := json.Unmarshal(b, &f); err != nil {
		return err
	}
	*v = jsonNumber(f)
	return nil
}
