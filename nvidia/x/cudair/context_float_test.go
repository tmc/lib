package cudair_test

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

type floatContextEmitter struct {
	fullEmitter
	body     string
	backward bool
}

func (n *floatContextEmitter) EmitForward(name string) (string, error) {
	return cudair.EmitFromBody(n, name, n.body)
}
func (n *floatContextEmitter) EmitBackward(name string, threads int) (string, error) {
	return cudair.EmitBackwardFromBody(n, name, n.body)
}

func TestContextFloatAccessAdmission(t *testing.T) {
	for _, c := range []struct {
		name, body     string
		backward, want bool
	}{
		{"first", "bool ok = __cudair_ctx.StoreFloat(0, 1.0f); (void)ok; return input;", false, true},
		{"last", "bool ok = __cudair_ctx.StoreFloat(4, 1.0f); (void)ok; return input;", false, true},
		{"misaligned", "bool ok = __cudair_ctx.StoreFloat(1, 1.0f); (void)ok; return input;", false, false},
		{"partial span", "bool ok = __cudair_ctx.StoreFloat(6, 1.0f); (void)ok; return input;", false, false},
		{"negative", "bool ok = __cudair_ctx.StoreFloat(-4, 1.0f); (void)ok; return input;", false, false},
		{"overflow span", "bool ok = __cudair_ctx.StoreFloat(2147483647, 1.0f); (void)ok; return input;", false, false},
		{"readonly", "bool ok = __cudair_ctx.StoreFloat(0, 1.0f); (void)ok;", true, false},
		{"read", "float saved; bool ok = __cudair_ctx.LoadFloat(4, &saved); (void)ok;", true, true},
		{"parameter destination", "bool ok = __cudair_ctx.LoadFloat(0, params); (void)ok;", true, false},
		{"cast destination", "bool ok = __cudair_ctx.LoadFloat(0, const_cast<float*>(params)); (void)ok;", true, false},
		{"wrong type", "uint8_t saved; bool ok = __cudair_ctx.LoadFloat(0, &saved); (void)ok;", true, false},
		{"unbounded helper", "float saved; bool ok = LoadFloat(0, &saved); (void)ok;", true, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			n := &floatContextEmitter{fullEmitter: fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{inWidth: 1, outWidth: 1, paddedWidth: 1, fwdBytes: 8}}, body: c.body}
			var e error
			if c.backward {
				_, e = cudair.EmitBackwardTree(n, "float_access", 32, cudair.GradientModeOverwrite)
			} else {
				_, e = cudair.EmitTree(n, "float_access")
			}
			if c.want && e != nil || !c.want && !errors.Is(e, cudair.ErrContextBoundsExceeded) {
				t.Fatalf("admission error=%v want allowed=%t", e, c.want)
			}
		})
	}
	if _, e := cudair.EmitBackwardTree(&fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{inWidth: 1, outWidth: 1}}, "bad_mode", 32, 99); e == nil {
		t.Fatal("invalid mode accepted")
	}
}

func TestContextFloatDynamicBoundsExecuteGeneratedBody(t *testing.T) {
	compiler, e := exec.LookPath("clang++")
	if e != nil {
		compiler, e = exec.LookPath("c++")
	}
	if e != nil {
		t.Skip("generated accessor execution needs a C++ compiler")
	}
	for _, offset := range []int{-4, 0, 1, 4, 5, 8, 2147483647} {
		t.Run(fmt.Sprint(offset), func(t *testing.T) {
			n := &floatContextEmitter{fullEmitter: fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{inWidth: 1, outWidth: 1, paddedWidth: 1, fwdBytes: 8}}, body: fmt.Sprintf("int offset = %d; bool ok = __cudair_ctx.StoreFloat(offset, 3.5f); float value = 0.0f; bool loaded = __cudair_ctx.LoadFloat(offset, &value); tvec<float, 1> out; out[0] = (ok && loaded ? value : -1.0f); return out;", offset)}
			unit, e := cudair.EmitTree(n, "float_access")
			if e != nil {
				t.Fatal(e)
			}
			want := "-1.0f"
			if offset == 0 || offset == 4 {
				want = "3.5f"
			}
			source := `#include <cstdint>
#define __device__
static constexpr int WARP_SIZE=32;
int lane_id(){return 0;}
template<typename T,int N>struct tvec{T data[N];T& operator[](int i){return data[i];}const T& operator[](int i)const{return data[i];}};
` + unit.Source + fmt.Sprintf("\nint main(){uint8_t bytes[16]; for(int i=0;i<16;++i)bytes[i]=0xa5;tvec<float,1> in;in[0]=0;auto out=float_access(in,nullptr,bytes+4);if(out[0]!=%s)return 2;for(int i=0;i<4;++i)if(bytes[i]!=0xa5||bytes[i+12]!=0xa5)return 3;return 0;}\n", want)
			dir := t.TempDir()
			file := filepath.Join(dir, "float.cc")
			binary := filepath.Join(dir, "float")
			if e := os.WriteFile(file, []byte(source), 0600); e != nil {
				t.Fatal(e)
			}
			if out, e := exec.Command(compiler, "-std=c++17", file, "-o", binary).CombinedOutput(); e != nil {
				t.Fatalf("compile: %v %s", e, out)
			}
			if out, e := exec.Command(binary).CombinedOutput(); e != nil {
				t.Fatalf("execute: %v %s", e, out)
			}
		})
	}
}
