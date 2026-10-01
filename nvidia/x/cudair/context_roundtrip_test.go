package cudair_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

type roundtripLeaf struct {
	fullEmitter
	bias int
}

func (n *roundtripLeaf) EmitForward(name string) (string, error) {
	return cudair.EmitFromBody(n, name, fmt.Sprintf("bool ok = __cudair_ctx.StoreByte(0, static_cast<uint8_t>(input[0] + %d)); (void)ok; return input;", n.bias))
}

func (n *roundtripLeaf) EmitBackward(name string, nThreads int) (string, error) {
	return cudair.EmitBackwardFromBody(n, name, "uint8_t saved; bool ok = __cudair_ctx.LoadByte(0, &saved); if (ok && dL_dx) { (*dL_dx)[0] += saved; }")
}

type roundtripParent struct {
	fullEmitter
	children []cudair.Emitter
}

func (n *roundtripParent) Children() []cudair.Emitter { return n.children }
func (n *roundtripParent) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	return cudair.EmitFromBody(n, name, fmt.Sprintf("auto value = %s(input, params, %s); return %s(value, params, %s);", children[0].Name, children[0].FwdCtxArg(), children[1].Name, children[1].FwdCtxArg()))
}

func (n *roundtripParent) EmitBackwardWithChildren(name string, children []cudair.ChildSymbol, nThreads int, mode cudair.GradientMode) (string, error) {
	var body strings.Builder
	for _, child := range children {
		fmt.Fprintf(&body, "%s(dL_dy, params, %s, dL_dparams, dL_dx);\n", child.Name, child.FwdCtxArg())
	}
	return cudair.EmitBackwardFromBody(n, name, body.String())
}

// This executes the generated bodies as host C++; CUDA device qualifiers and
// lane_id are supplied by the harness. It is not GPU qualification.
func TestContextRoundTripGeneratedBodies(t *testing.T) {
	compilers := []string{os.Getenv("CXX"), "clang++", "c++", "g++"}
	var compiler string
	var tried []string
	for _, candidate := range compilers {
		if candidate == "" {
			continue
		}
		tried = append(tried, candidate)
		if path, err := exec.LookPath(candidate); err == nil {
			compiler = path
			break
		}
	}
	if compiler == "" {
		t.Skipf("generated context round-trip requires a C++ compiler; tried %s", strings.Join(tried, ", "))
	}
	base := forwardOnlyEmitter{inWidth: 1, outWidth: 1, paddedWidth: 1, fwdBytes: 1, aligned: true}
	first, second := base, base
	first.name, second.name = "first_context", "second_context"
	root := &roundtripParent{fullEmitter: fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{
		name: "context_root", inWidth: 1, outWidth: 1, paddedWidth: 1}}, children: []cudair.Emitter{
		&roundtripLeaf{fullEmitter: fullEmitter{forwardOnlyEmitter: first}, bias: 3},
		&roundtripLeaf{fullEmitter: fullEmitter{forwardOnlyEmitter: second}, bias: 9},
	}}
	forward, err := cudair.EmitTree(root, "write_context")
	if err != nil {
		t.Fatal(err)
	}
	backward, err := cudair.EmitBackwardTree(root, "read_context", 32, cudair.GradientModeIgnore)
	if err != nil {
		t.Fatal(err)
	}
	if forward.FwdCtxBytes != 2 || backward.FwdCtxBytes != 2 {
		t.Fatalf("context bytes = %d/%d, want two child regions", forward.FwdCtxBytes, backward.FwdCtxBytes)
	}
	const stride = "fwd_ctx += lane_id() * 1;"
	if strings.Count(forward.Source, stride) != 2 || strings.Count(backward.Source, stride) != 2 {
		t.Fatal("missing production per-lane child preambles")
	}
	for _, tc := range []struct {
		name, forward, backward string
		wantDisagreement        bool
	}{
		{"baseline", forward.Source, backward.Source, false},
		{"forward stride dropped", strings.ReplaceAll(forward.Source, stride, "fwd_ctx += 0;"), backward.Source, true},
		{"backward stride dropped", forward.Source, strings.ReplaceAll(backward.Source, stride, "fwd_ctx += 0;"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			directory := t.TempDir()
			source := `#include <cstdint>
#include <cstdio>
#define __device__
static constexpr int WARP_SIZE = 32;
static int current_lane;
static int lane_id() { return current_lane; }
template<typename T, int N> struct tvec {
    T data[N];
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
};
` + tc.forward + "\n" + tc.backward + `
int main() {
    uint8_t context[64] = {};
    for (current_lane = 0; current_lane < WARP_SIZE; ++current_lane) {
        tvec<float,1> input = {{float(current_lane + 1)}};
        write_context(input, nullptr, context);
    }
    for (current_lane = 0; current_lane < WARP_SIZE; ++current_lane) {
        tvec<float,1> dy = {{0}}, dx = {{0}};
        read_context(dy, nullptr, context, nullptr, &dx);
        float expected = 2 * (current_lane + 1) + 12;
        if (dx[0] != expected) {
            std::fprintf(stderr, "context disagreement lane %d: %g != %g\n", current_lane, dx[0], expected);
            return 2;
        }
    }
    return 0;
}
`
			file := filepath.Join(directory, "context.cc")
			if err := os.WriteFile(file, []byte(source), 0600); err != nil {
				t.Fatal(err)
			}
			binary := filepath.Join(directory, "context")
			if output, err := exec.Command(compiler, "-std=c++17", file, "-o", binary).CombinedOutput(); err != nil {
				t.Fatalf("compile generated bodies: %v\n%s", err, output)
			}
			output, err := exec.Command(binary).CombinedOutput()
			if tc.wantDisagreement {
				if failure, ok := err.(*exec.ExitError); !ok || failure.ExitCode() != 2 || !strings.Contains(string(output), "context disagreement lane") {
					t.Fatalf("stride mutant result = %v, %s, want runtime context disagreement", err, output)
				}
			} else if err != nil {
				t.Fatalf("generated round-trip: %v\n%s", err, output)
			}
		})
	}
}
