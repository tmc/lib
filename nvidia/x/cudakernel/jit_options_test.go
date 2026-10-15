package cudakernel_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudakernel"
)

func TestJITPresenceDriverSerialization(t *testing.T) {
	for _, tc := range []struct {
		name    string
		jit     cudakernel.JITOptions
		present bool
		level   uintptr
	}{
		{"unset", cudakernel.JITOptions{}, false, 0},
		{"zero", cudakernel.JITOptions{OptLevelSet: true}, true, 0},
		{"legacy positive", cudakernel.JITOptions{OptLevel: 4}, true, 4},
		{"marked positive", cudakernel.JITOptions{OptLevel: 4, OptLevelSet: true}, true, 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.jit.Validate(); err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(tc.jit)
			if err != nil {
				t.Fatal(err)
			}
			var decoded cudakernel.JITOptions
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatal(err)
			}
			if decoded != tc.jit {
				t.Fatal(decoded, tc.jit)
			}
			opts, vals := decoded.DriverOptions()
			if !tc.present {
				if len(opts) != 0 || len(vals) != 0 {
					t.Fatal(opts, vals)
				}
			} else if !reflect.DeepEqual(opts, []driver.JITOption{driver.JITOptimizationLevel}) || !reflect.DeepEqual(vals, []uintptr{tc.level}) {
				t.Fatal(opts, vals)
			}
		})
	}
	var old cudakernel.JITOptions
	if err := json.Unmarshal([]byte(`{"opt_level":0}`), &old); err != nil || old.OptLevelSet {
		t.Fatal(old, err)
	}
	if opts, _ := old.DriverOptions(); len(opts) != 0 {
		t.Fatal("old zero changed meaning")
	}
}

func TestJITInvalidAdmissionBeforeLoad(t *testing.T) {
	jit := cudakernel.JITOptions{OptLevel: 5}
	if jit.Validate() == nil {
		t.Fatal("invalid level admitted")
	}
	// A hand-authored descriptor fixture, never compiled or loaded.
	art, err := cudakernel.NewArtifact(cudakernel.KindPTX, []byte("fixture ptx"), "k", cudakernel.Schema{}, "compute_80")
	if err != nil {
		t.Fatal(err)
	}
	art.JITOptions = jit
	if art.Validate() == nil {
		t.Fatal("invalid descriptor admitted")
	}
	_, err = art.Load(nil)
	if err == nil || !strings.Contains(err.Error(), "optimization level") {
		t.Fatal("validation did not precede load", err)
	}
}
