package cudaprobe

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type fixtureResolver struct {
	result Resolution
	err    error
}

func (r fixtureResolver) Resolve(context.Context, Target) (Resolution, error) { return r.result, r.err }

func TestSymbolResolution(t *testing.T) {
	target := Target{Library: "/fixture/libcuda.so.1", Symbol: "cuLaunchKernel", SymbolVersion: "fixture-v2"}
	valid := Resolution{Target: target, BuildID: "synthetic-build", Offset: 64, FileSize: 128, Function: true}
	for _, tt := range []struct {
		name   string
		change func(*Resolution)
		err    error
		wantOK bool
	}{
		{"versioned", func(*Resolution) {}, nil, true},
		{"plt", func(r *Resolution) { r.PLT = true }, nil, false},
		{"not-function", func(r *Resolution) { r.Function = false }, nil, false},
		{"end-offset", func(r *Resolution) { r.Offset = r.FileSize }, nil, false},
		{"wrong-version", func(r *Resolution) { r.Target.SymbolVersion = "old" }, nil, false},
		{"wrong-file", func(r *Resolution) { r.Target.Library = "other.so" }, nil, false},
		{"no-build-id", func(r *Resolution) { r.BuildID = "" }, nil, false},
		{"missing-symbol", func(*Resolution) {}, errors.New("symbol absent"), false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := valid
			tt.change(&r)
			got, err := Resolve(context.Background(), fixtureResolver{r, tt.err}, target)
			if (err == nil) != tt.wantOK {
				t.Fatalf("resolution=%+v error=%v", got, err)
			}
			if err != nil && !strings.Contains(err.Error(), target.Symbol) {
				t.Fatal("error omitted symbol")
			}
			if tt.wantOK && got != valid {
				t.Fatalf("got=%+v", got)
			}
		})
	}
	a := Targets()
	a[0].Symbol = "changed"
	if Targets()[0].Symbol == "changed" {
		t.Fatal("shared target table")
	}
}
