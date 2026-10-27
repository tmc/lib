package cudair_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudair"
)

type contextSizeEmitter struct {
	bytes    int
	children []cudair.Emitter
	seen     []cudair.ChildSymbol
	emitted  int
	ctxErr   error
}

func (n *contextSizeEmitter) Name() string                                    { return "context_size" }
func (n *contextSizeEmitter) InputWidth() int                                 { return 1 }
func (n *contextSizeEmitter) OutputWidth() int                                { return 1 }
func (n *contextSizeEmitter) PaddedOutputWidth() int                          { return 1 }
func (n *contextSizeEmitter) FwdCtxBytes() (int, error)                       { return n.bytes, n.ctxErr }
func (n *contextSizeEmitter) FwdCtxAlignedPerElement() bool                   { return false }
func (n *contextSizeEmitter) Children() []cudair.Emitter                      { return n.children }
func (n *contextSizeEmitter) BackwardShmemBytes(int, cudair.GradientMode) int { return 0 }

func (n *contextSizeEmitter) EmitForward(name string) (string, error) {
	n.emitted++
	return cudair.EmitFromBody(n, name, "\treturn input;")
}

func (n *contextSizeEmitter) EmitBackward(name string, threads int) (string, error) {
	n.emitted++
	return cudair.EmitBackwardFromBody(n, name, "\t// no gradient writes")
}

func (n *contextSizeEmitter) EmitForwardWithChildren(name string, children []cudair.ChildSymbol) (string, error) {
	n.seen = append([]cudair.ChildSymbol(nil), children...)
	return n.EmitForward(name)
}

func (n *contextSizeEmitter) EmitBackwardWithChildren(name string, children []cudair.ChildSymbol, threads int, mode cudair.GradientMode) (string, error) {
	n.seen = append([]cudair.ChildSymbol(nil), children...)
	return n.EmitBackward(name, threads)
}

func emitContextSizeTree(n *contextSizeEmitter, backward bool) (cudair.Unit, error) {
	if backward {
		return cudair.EmitBackwardTree(n, "context_root", 32, cudair.GradientModeIgnore)
	}
	return cudair.EmitTree(n, "context_root")
}

func TestTreeContextSizeBounds(t *testing.T) {
	for _, backward := range []bool{false, true} {
		direction := "forward"
		if backward {
			direction = "backward"
		}
		t.Run(direction, func(t *testing.T) {
			for _, tt := range []struct {
				name     string
				parent   int
				children []int
				want     int
				overflow bool
				negative bool
			}{
				{name: "child sum overflow", children: []int{math.MaxInt, 1}, overflow: true},
				{name: "parent sum overflow", parent: 1, children: []int{math.MaxInt}, overflow: true},
				{name: "parent and children overflow", parent: math.MaxInt, children: []int{0, 1}, overflow: true},
				{name: "exact child limit", children: []int{math.MaxInt - 1, 1}, want: math.MaxInt},
				{name: "exact parent limit", parent: 1, children: []int{math.MaxInt - 1}, want: math.MaxInt},
				{name: "leaf limit", parent: math.MaxInt, want: math.MaxInt},
				{name: "negative parent", parent: -1, negative: true},
				{name: "negative child", children: []int{-1}, negative: true},
				{name: "ordinary regions", parent: 4, children: []int{3, 5}, want: 12},
				{name: "empty context", children: []int{0, 0}},
			} {
				t.Run(tt.name, func(t *testing.T) {
					n := &contextSizeEmitter{bytes: tt.parent}
					for _, size := range tt.children {
						n.children = append(n.children, &contextSizeEmitter{bytes: size})
					}
					unit, err := emitContextSizeTree(n, backward)
					if tt.overflow || tt.negative {
						if err == nil {
							t.Fatalf("context total = %d, error = nil; want refusal", unit.FwdCtxBytes)
						}
						if tt.overflow && !errors.Is(err, cudair.ErrContextBoundsExceeded) {
							t.Fatalf("error = %v, want ErrContextBoundsExceeded", err)
						}
						if tt.negative && !strings.Contains(err.Error(), "negative fwd ctx bytes") {
							t.Fatalf("error = %v, want negative context refusal", err)
						}
						if unit.Source != "" || unit.Symbols != nil || unit.FwdCtxBytes != 0 || tt.overflow && n.emitted != 0 {
							t.Fatalf("invalid tree published output: unit=%+v, root emissions=%d", unit, n.emitted)
						}
						return
					}
					if err != nil {
						t.Fatal(err)
					}
					if unit.FwdCtxBytes != tt.want || n.emitted != 1 || len(unit.Symbols) != len(tt.children)+1 {
						t.Fatalf("context total=%d, emissions=%d, symbols=%v; want total=%d and one root", unit.FwdCtxBytes, n.emitted, unit.Symbols, tt.want)
					}
					offset := 0
					for i, child := range n.seen {
						if child.FwdCtxOffset != offset || child.ParentFwdCtxBytes != tt.parent {
							t.Fatalf("child %d region = %+v, want offset=%d parent=%d", i, child, offset, tt.parent)
						}
						offset += tt.children[i]
					}
				})
			}
		})
	}
}

func TestTreeContextSizeNestedAndErrors(t *testing.T) {
	for _, backward := range []bool{false, true} {
		name := "forward"
		if backward {
			name = "backward"
		}
		t.Run(name, func(t *testing.T) {
			n := &contextSizeEmitter{children: []cudair.Emitter{
				&contextSizeEmitter{bytes: 1, children: []cudair.Emitter{&contextSizeEmitter{bytes: math.MaxInt}}},
			}}
			if _, err := emitContextSizeTree(n, backward); !errors.Is(err, cudair.ErrContextBoundsExceeded) {
				t.Fatalf("nested overflow error = %v", err)
			}
			want := errors.New("context extent unavailable")
			n = &contextSizeEmitter{children: []cudair.Emitter{&contextSizeEmitter{ctxErr: want}}}
			if _, err := emitContextSizeTree(n, backward); !errors.Is(err, want) {
				t.Fatalf("child error = %v, want %v", err, want)
			}
		})
	}
}
