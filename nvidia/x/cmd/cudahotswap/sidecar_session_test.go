package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/egraph"
)

type sidecarTestLease struct {
	fail   bool
	closed int
}

func (l *sidecarTestLease) Close() error {
	l.closed++
	if l.fail {
		return errors.New("release failed")
	}
	return nil
}

type sidecarTestPending struct {
	fail   bool
	waited int
}

func (p *sidecarTestPending) Wait(context.Context) error {
	p.waited++
	if p.fail {
		return errors.New("wait failed")
	}
	return nil
}

func TestSidecarInputsStrict(t *testing.T) {
	for _, input := range []string{`null`, `[]`, `{"a":[1],"a":[2]}`, `{"a":null}`, `{"a":[null]}`, `{"a":["1"]}`, `{"a":[1e50]}`, `{"a":[]}`, `{"a":[1]} {}`, `{"a":[1],"": [2]}`} {
		if _, err := decodeSidecarInputs([]byte(input)); err == nil {
			t.Fatalf("accepted %s", input)
		}
	}
	if _, err := decodeSidecarInputs([]byte(`{"a":[1,-2,0.5]}`)); err != nil {
		t.Fatal(err)
	}
	if _, err := decodeSidecarInputs([]byte(`{"a":[` + strings.Repeat("0,", 65536) + `0]}`)); err == nil {
		t.Fatal("element bound ignored")
	}
	var many strings.Builder
	many.WriteByte('{')
	for i := 0; i < 65; i++ {
		if i > 0 {
			many.WriteByte(',')
		}
		many.WriteString(`"` + strings.Repeat("a", i+1) + `":[0]`)
	}
	many.WriteByte('}')
	if _, err := decodeSidecarInputs([]byte(many.String())); err == nil {
		t.Fatal("parameter bound ignored")
	}
}

func TestSidecarPointerABI(t *testing.T) {
	r := &sidecarReference{source: egraph.Source{CUDA: `extern "C" __global__ void chain(const float* a, const float* b, float* eg_output) {}`, Parameters: []egraph.Parameter{{Name: "a"}, {Name: "b"}}}}
	if err := sidecarABI(r, "chain"); err != nil {
		t.Fatal(err)
	}
	for _, code := range []string{
		`// extern "C" __global__ void chain(const float* a, const float* b, float* eg_output) {}`,
		`extern "C" __global__ void chain(const float* b, const float* a, float* eg_output) {}`,
		`extern "C" __global__ void chain(const float* a, const float* b, float* eg_output, unsigned n) {}`,
		`#define float int
extern "C" __global__ void chain(const float* a, const float* b, float* eg_output) {}`,
		`extern "C" __global__ void chain(float* a, const float* b, float* eg_output) {}`,
	} {
		copy := *r
		copy.source = r.source
		copy.source.CUDA = code
		if err := sidecarABI(&copy, "chain"); err == nil {
			t.Fatal("ABI mutation accepted", code)
		}
	}
}

func TestSidecarCheckCompletionAndPoison(t *testing.T) {
	p := new(sidecarTestPending)
	var output []float32
	leases := []*sidecarTestLease{}
	a := &sidecarApplication{owners: &sessionOwners{}, ref: &sidecarReference{want: []float32{8, 14}}}
	a.owners.acquire = func(context.Context, string) (io.Closer, error) {
		l := new(sidecarTestLease)
		leases = append(leases, l)
		return l, nil
	}
	a.write = func(v []float32) error { output = append([]float32(nil), v...); return nil }
	a.read = func(v []float32) error {
		if p.waited == 0 {
			t.Fatal("read before completion")
		}
		copy(v, output)
		return nil
	}
	missing := false
	inputs := []float32{1, 3}
	restored := 0
	a.restore = func() error { inputs = []float32{1, 3}; restored++; return nil }
	a.submit = func(context.Context) (sidecarPending, error) {
		if len(inputs) != 2 || inputs[0] != 1 || inputs[1] != 3 {
			t.Fatal("launch used edited inputs")
		}
		inputs[0] = 999
		if !math.IsNaN(float64(output[0])) || !math.IsNaN(float64(output[1])) {
			t.Fatal("no poisoned prefill")
		}
		output[0] = 8
		if !missing {
			output[1] = 14
		}
		return p, nil
	}
	if err := a.check(context.Background(), cudahotswap.State{}); err != nil {
		t.Fatal(err)
	}
	if inputs[0] != 999 || restored != 1 {
		t.Fatal("input edit control did not execute")
	}
	if a.pending != nil || a.readLease != nil || p.waited != 1 || len(leases) != 2 || leases[0].closed != 1 || leases[1].closed != 1 {
		t.Fatal("completion/lease ownership changed")
	}
	missing = true
	if err := a.check(context.Background(), cudahotswap.State{}); err == nil {
		t.Fatal("missing store retained previous output")
	}
	p.fail = true
	if err := a.check(context.Background(), cudahotswap.State{}); err == nil || a.pending != p {
		t.Fatal("failed wait lost owner")
	}
	p.fail = false
	if err := a.finish(context.Background()); err != nil || a.pending != nil {
		t.Fatal(err)
	}
}

func TestSidecarRetainsReadOwner(t *testing.T) {
	l := &sidecarTestLease{fail: true}
	a := &sidecarApplication{readLease: l}
	if err := a.releaseRead(); err == nil || a.readLease != l {
		t.Fatal("failed lease release lost owner")
	}
	l.fail = false
	if err := a.releaseRead(); err != nil || a.readLease != nil {
		t.Fatal(err)
	}
}

func TestSidecarCLIRequiresExplicitFlags(t *testing.T) {
	for _, args := range [][]string{
		{"-watch", "x.cu", "-sidecar", "x.json"},
		{"-watch", "x.cu", "-inputs", "i.json"},
		{"-watch", "x.cu", "-sidecar", "x.json", "-inputs", "i.json"},
		{"-watch", "x.cu", "-sidecar", "x.json", "-inputs", "i.json", "-entry", "chain", "-n", "32"},
	} {
		called := false
		err := run(context.Background(), args, &bytes.Buffer{}, &bytes.Buffer{}, func(context.Context, config) (*cudahotswap.Service, func() error, error) {
			called = true
			return nil, nil, errors.New("unexpected setup")
		})
		if err == nil || called {
			t.Fatal("invalid sidecar flags admitted setup")
		}
	}
}

func FuzzSidecarInputs(f *testing.F) {
	f.Add([]byte(`{"a":[1,2]}`))
	f.Add([]byte(`{"a":[1],"a":[2]}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 8<<20 {
			return
		}
		m, err := decodeSidecarInputs(data)
		if err != nil {
			return
		}
		if len(m) > 64 {
			t.Fatal("parameter bound")
		}
		total := 0
		for _, v := range m {
			total += len(v)
			for _, x := range v {
				if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
					t.Fatal("nonfinite")
				}
			}
		}
		if total > 65536 {
			t.Fatal("element bound")
		}
	})
}
