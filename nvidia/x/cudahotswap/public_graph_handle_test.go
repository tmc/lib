package cudahotswap

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
)

// publicGraphHandle returns a captured native identity only while the exact
// executable remains retained by this fixture. It does not retain topology.
func publicGraphHandle(handles map[*cudagraph.Exec]driver.GraphExec, exec *cudagraph.Exec, owners ...[]*cudagraph.Exec) (driver.GraphExec, error) {
	if exec == nil {
		return 0, errors.New("graph receipt: missing executable owner")
	}
	handle, ok := handles[exec]
	if !ok || handle == 0 {
		return 0, errors.New("graph receipt: missing captured executable identity")
	}
	for _, group := range owners {
		for _, owner := range group {
			if owner == exec {
				return handle, nil
			}
		}
	}
	return 0, errors.New("graph receipt: executable is no longer retained")
}

func TestPublicGraphCapturedHandleOwnership(t *testing.T) {
	a, b := new(cudagraph.Exec), new(cudagraph.Exec)
	for _, tt := range []struct {
		name    string
		exec    *cudagraph.Exec
		handles map[*cudagraph.Exec]driver.GraphExec
		owners  [][]*cudagraph.Exec
		want    driver.GraphExec
	}{
		{"registered", a, map[*cudagraph.Exec]driver.GraphExec{a: 7}, [][]*cudagraph.Exec{{a}, {b}}, 7},
		{"raw", b, map[*cudagraph.Exec]driver.GraphExec{b: 9}, [][]*cudagraph.Exec{{a}, {b}}, 9},
		{"nil", nil, map[*cudagraph.Exec]driver.GraphExec{a: 7}, [][]*cudagraph.Exec{{a}}, 0},
		{"absent mapping", b, map[*cudagraph.Exec]driver.GraphExec{a: 7}, [][]*cudagraph.Exec{{b}}, 0},
		{"zero mapping", a, map[*cudagraph.Exec]driver.GraphExec{a: 0}, [][]*cudagraph.Exec{{a}}, 0},
		{"retired owner", a, map[*cudagraph.Exec]driver.GraphExec{a: 7}, [][]*cudagraph.Exec{{b}}, 0},
		{"empty owners", a, map[*cudagraph.Exec]driver.GraphExec{a: 7}, nil, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := publicGraphHandle(tt.handles, tt.exec, tt.owners...)
			if got != tt.want || (err == nil) != (tt.want != 0) {
				t.Fatalf("got %d,%v, want %d", got, err, tt.want)
			}
		})
	}
}
