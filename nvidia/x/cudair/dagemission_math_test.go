package cudair

import (
	"fmt"
	"testing"
)

type pureMathTestEmitter struct{ body string }

func (n *pureMathTestEmitter) InputWidth() int               { return 1 }
func (n *pureMathTestEmitter) OutputWidth() int              { return 1 }
func (n *pureMathTestEmitter) PaddedOutputWidth() int        { return 1 }
func (n *pureMathTestEmitter) FwdCtxBytes() (int, error)     { return 0, nil }
func (n *pureMathTestEmitter) FwdCtxAlignedPerElement() bool { return false }
func (n *pureMathTestEmitter) EmitForward(name string) (string, error) {
	return EmitFromBody(n, name, n.body)
}

func TestPureCUDAUnaryMathValueCalls(t *testing.T) {
	for _, name := range []string{"expf", "erff", "sqrtf", "rsqrtf"} {
		n := &pureMathTestEmitter{body: fmt.Sprintf("float v = %s(input[0]); tvec<float, 1> out; out[0] = v; return out;", name)}
		if _, err := EmitTree(n, "leaf"); err != nil {
			t.Fatal(name, err)
		}
		for _, arg := range []string{"", "input[0], input[0]", "fwd_ctx", "params", "input", "&input[0]", "nullptr"} {
			n.body = fmt.Sprintf("float v = %s(%s); return input;", name, arg)
			if _, err := EmitTree(n, "leaf"); err == nil {
				t.Fatal("invalid math argument admitted", name, arg)
			}
		}
		for _, body := range []string{
			fmt.Sprintf("float v = ns::%s(input[0]); return input;", name),
			fmt.Sprintf("float v = object.%s(input[0]); return input;", name),
			fmt.Sprintf("float %s; float v = %s(input[0]); return input;", name, name),
		} {
			n.body = body
			if _, err := EmitTree(n, "leaf"); err == nil {
				t.Fatal("qualified/shadowed math admitted", body)
			}
		}
	}
	n := &pureMathTestEmitter{body: "float v = unknown(input[0]); return input;"}
	if _, err := EmitTree(n, "leaf"); err == nil {
		t.Fatal("unknown callee admitted")
	}
	for _, body := range []string{"unknown(input[0]); return input;", "expf(); return input;", "erff(input[0], input[0]); return input;"} {
		n.body = body
		if _, err := EmitTree(n, "leaf"); err == nil {
			t.Fatal("leading call bypassed validation", body)
		}
	}
}
