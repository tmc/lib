package cudair_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

func requireContextRefusal(t *testing.T, err error, node, reason string) {
	t.Helper()
	if !errors.Is(err, cudair.ErrContextBoundsExceeded) || !strings.Contains(err.Error(), node) || !strings.Contains(err.Error(), reason) {
		t.Fatalf("error = %v, want named ErrContextBoundsExceeded with %q", err, reason)
	}
}

func TestContextCheckerRefusedForms(t *testing.T) {
	for _, backward := range []bool{false, true} {
		direction := "forward"
		if backward {
			direction = "backward"
		}
		for _, tc := range []struct {
			name, body, reason string
			backwardOnly       bool
		}{
			{"call", "float v = 1; atomicAdd(params, v);", "call is outside the generated write allowlist", false},
			{"increment", "++params[0];", "increment target is not a body local", false},
			{"decrement", "--params[0];", "increment target is not a body local", false},
			{"alias params", "auto p = params; p[0] = 1;", "auto local aliases a pointer parameter", false},
			{"alias input", "auto p = input; p[0] = 1;", "auto local aliases a pointer parameter", false},
			{"alias upstream", "auto p = dL_dy; p[0] = 1;", "auto local aliases a pointer parameter", true},
			{"store", "bool ok = __cudair_ctx.StoreByte(0, 1);", "backward context is read-only", true},
			{"unsupported local", "half saved; saved = 1;", "unsupported declaration half saved", true},
			{"unknown write", "saved[0] = 1;", "write target saved is not a body local", true},
			{"load bound", "uint8_t saved; bool ok = __cudair_ctx.LoadByte(4, &saved);", "constant context offset is outside the declared region", false},
			{"directive", "#define A 1\n", "preprocessor directive", false},
			{"paste", "a##b;", "unsupported source splice", false},
			{"splice", "a\\\nb;", "unsupported source splice", false},
		} {
			if tc.backwardOnly && !backward {
				continue
			}
			t.Run(direction+"/"+tc.name, func(t *testing.T) {
				base := forwardOnlyEmitter{name: "refused_node", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 4}
				var err error
				if backward {
					node := &gradientWriter{fullEmitter: fullEmitter{forwardOnlyEmitter: base}, body: tc.body}
					_, err = cudair.EmitBackwardTree(node, "checked", 32, cudair.GradientModeIgnore)
				} else {
					node := &contextWriter{forwardOnlyEmitter: base, emit: tc.body + " return input;"}
					_, err = cudair.EmitTree(node, "checked")
				}
				requireContextRefusal(t, err, base.name, tc.reason)
			})
		}
	}
}

type checkedComposite struct {
	fullEmitter
	children []cudair.Emitter
	body     string
}

func (c *checkedComposite) Children() []cudair.Emitter { return c.children }
func (c *checkedComposite) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	return cudair.EmitFromBody(c, name, fmt.Sprintf(c.body, children[0].Name))
}
func (c *checkedComposite) EmitBackwardWithChildren(name string, children []cudair.ChildSymbol, nThreads int, mode cudair.GradientMode) (string, error) {
	return cudair.EmitBackwardFromBody(c, name, fmt.Sprintf(c.body, children[0].Name))
}

func TestContextCheckerCompositeRefusedForms(t *testing.T) {
	for _, backward := range []bool{false, true} {
		direction := "forward"
		if backward {
			direction = "backward"
		}
		for _, tc := range []struct{ name, body, reason string }{
			{"shadow", "float %s;", "local shadows an emitted child"},
			{"qualified", "ns::%s(input, params, fwd_ctx);", "context parameter used outside an exact child call"},
			{"argument count", "%s(input, params, fwd_ctx, input);", "context parameter used outside an exact child call"},
			{"parent write", "params[0] = 1; // %s", "write target is a parameter or context pointer"},
		} {
			t.Run(direction+"/"+tc.name, func(t *testing.T) {
				base := forwardOnlyEmitter{name: "composite_guard", inWidth: 16, outWidth: 16, paddedWidth: 16}
				child := &fullEmitter{forwardOnlyEmitter: forwardOnlyEmitter{name: "child", inWidth: 16, outWidth: 16, paddedWidth: 16}}
				node := &checkedComposite{fullEmitter: fullEmitter{forwardOnlyEmitter: base}, children: []cudair.Emitter{child}, body: tc.body}
				if backward && tc.name == "qualified" {
					node.body = "ns::%s(dL_dy, params, fwd_ctx, dL_dparams, dL_dx);"
				}
				var err error
				if backward {
					_, err = cudair.EmitBackwardTree(node, "checked", 32, cudair.GradientModeIgnore)
				} else {
					_, err = cudair.EmitTree(node, "checked")
				}
				requireContextRefusal(t, err, base.name, tc.reason)
			})
		}
	}
}

func TestContextCheckerRuntimeBounds(t *testing.T) {
	base := forwardOnlyEmitter{name: "runtime_bound", inWidth: 16, outWidth: 16, paddedWidth: 16, fwdBytes: 4}
	forward, err := cudair.EmitTree(&contextWriter{forwardOnlyEmitter: base, emit: "return input;"}, "checked")
	if err != nil {
		t.Fatal(err)
	}
	backward, err := cudair.EmitBackwardTree(&fullEmitter{forwardOnlyEmitter: base}, "checked", 32, cudair.GradientModeIgnore)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, source string
		count        int
	}{
		{"forward", forward.Source, 2}, {"backward", backward.Source, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := strings.Count(tc.source, "i < 0 || i >= n || p == nullptr"); got != tc.count {
				t.Fatalf("runtime bounds count = %d, want %d:\n%s", got, tc.count, tc.source)
			}
		})
	}
}
