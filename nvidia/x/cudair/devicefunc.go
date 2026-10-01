package cudair

import (
	"errors"
	"fmt"
	"regexp"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
)

// GradientMode specifies how gradients are accumulated into parameter buffers.
type GradientMode int

const (
	// GradientModeIgnore ignores gradient writes.
	GradientModeIgnore GradientMode = iota
	// GradientModeOverwrite overwrites gradient memory directly.
	GradientModeOverwrite
	// GradientModeAccumulate atomically adds gradients into memory.
	GradientModeAccumulate
)

// String returns the string representation of the gradient mode.
func (m GradientMode) String() string {
	switch m {
	case GradientModeIgnore:
		return "Ignore"
	case GradientModeOverwrite:
		return "Overwrite"
	case GradientModeAccumulate:
		return "Accumulate"
	default:
		return fmt.Sprintf("GradientMode(%d)", m)
	}
}

// Emitter is implemented by nodes that can render themselves as a self-contained CUDA __device__
// function. Nodes without this support must not implement it. Unsupported nodes return
// ErrEmissionUnsupported together with the node's name.
//
// Writable body locals support float, int, bool, uint8_t and auto, with one declaration per
// statement. Template values such as tvec<float, N> may be declared without an initializer, then
// written element by element. Template initializers and comma-separated declarations are refused.
// Other local types are refused when written. Pointer and reference locals are refused.
//
// Backward bodies may also write through generated dL_dx and dL_dparams output parameters.
// Input and context parameters remain read-only. Supported forms include (*dL_dx)[i],
// dL_dparams[i] and *(dL_dparams + i). A direct *dL_dparams assignment is refused; use an
// indexed output write.
//
// Context accessor integer constants and arithmetic offsets are checked at emission; dynamic
// offsets are checked against FwdCtxBytes at run time by LoadByte/StoreByte.
type Emitter interface {
	EmitForward(name string) (string, error)
	InputWidth() int
	OutputWidth() int
	PaddedOutputWidth() int
	FwdCtxBytes() (int, error)
	FwdCtxAlignedPerElement() bool
}

// BackwardEmitter extends Emitter for nodes that support backward device code generation.
type BackwardEmitter interface {
	Emitter
	EmitBackward(name string, nThreads int) (string, error)
	BackwardShmemBytes(nThreads int, mode GradientMode) int
}

// CompositeEmitter represents a sequential composition and exposes its ordered child emitters.
type CompositeEmitter interface {
	Emitter
	Children() []Emitter
}

// ChildSymbol carries the exact emitted child symbol and its context-region offset.
type ChildSymbol struct {
	Name                string
	FwdCtxOffset        int
	ParentFwdCtxBytes   int
	ParentFwdCtxAligned bool
}

// FwdCtxArg returns the parent expression for passing this child's context region while preserving
// the per-lane offset already applied by the parent.
func (c ChildSymbol) FwdCtxArg() string {
	if c.FwdCtxOffset == 0 && c.ParentFwdCtxBytes == 0 {
		return "fwd_ctx"
	}
	if c.ParentFwdCtxBytes == 0 {
		return fmt.Sprintf("(fwd_ctx ? fwd_ctx + WARP_SIZE * %d : nullptr)", c.FwdCtxOffset)
	}
	base := c.ParentFwdCtxBytes + c.FwdCtxOffset
	if c.ParentFwdCtxAligned {
		return fmt.Sprintf("(fwd_ctx ? fwd_ctx + WARP_SIZE * %d - lane_id() * %d : nullptr)", base, c.ParentFwdCtxBytes)
	}
	return fmt.Sprintf("(fwd_ctx ? fwd_ctx + WARP_SIZE * %d : nullptr)", base)
}

// CompositeSymbolEmitter emits a parent using the exact child symbols and context offsets selected
// by EmitTree.
type CompositeSymbolEmitter interface {
	EmitForwardWithChildren(name string, children []ChildSymbol) (string, error)
}

// BackwardCompositeSymbolEmitter emits a backward parent using names chosen by EmitTree.
type BackwardCompositeSymbolEmitter interface {
	EmitBackwardWithChildren(name string, children []ChildSymbol, nThreads int, mode GradientMode) (string, error)
}

// ShmemProvider allows an emitter to declare dynamic shared memory requirements.
type ShmemProvider interface {
	ShmemBytes() int
}

// TypedEmitter allows an emitter to customize C++ scalar and vector types.
type TypedEmitter interface {
	InputType() string
	ParamsType() string
	ComputeType() string
}

// ErrEmissionUnsupported reports that a node does not support forward device code generation.
var ErrEmissionUnsupported = fmt.Errorf("cudair: emission unsupported: %w", cudaerrors.ErrUnsupported)

// ErrBackwardEmissionUnsupported reports that a node does not support backward device code
// generation.
var ErrBackwardEmissionUnsupported = fmt.Errorf("cudair: backward emission unsupported: %w", cudaerrors.ErrUnsupported)

// ErrContextBoundsExceeded reports emitted code that violates a context or write rule, including
// out-of-region context access, unsupported calls and writes outside allowed body locals or
// backward gradient outputs.
var ErrContextBoundsExceeded = errors.New("cudair: emitted context or write rule violated")

// ValidateFwdContextBounds verifies that the actual context bytes written by node does not exceed
// its declared FwdCtxBytes.
func ValidateFwdContextBounds(node Emitter, bytesWritten int) error {
	decl, err := node.FwdCtxBytes()
	if err != nil {
		return err
	}
	if decl < 0 {
		return fmt.Errorf("cudair: node %s declares negative fwd ctx bytes %d", NodeName(node), decl)
	}
	if bytesWritten < 0 {
		return fmt.Errorf("cudair: node %s reports negative fwd ctx writes %d", NodeName(node), bytesWritten)
	}
	if bytesWritten > decl {
		return fmt.Errorf("%w: node %s wrote %d bytes exceeding declared %d bytes",
			ErrContextBoundsExceeded, NodeName(node), bytesWritten, decl)
	}
	return nil
}

// validateEmittedContextUse permits a node to pass its context pointer to an emitted child, but not
// to inspect or modify that pointer itself. The child owns its region and is checked by the same
// rule when it is emitted.
func validateEmittedContextUse(node Emitter, source string, children []ChildSymbol, preamble string) error {
	return validateGeneratedContextBody(node, source, children, preamble)
}

// NodeName returns a readable string identifier for node.
func NodeName(node any) string {
	if node == nil {
		return "<nil>"
	}
	type named interface {
		Name() string
	}
	if n, ok := node.(named); ok && n.Name() != "" {
		return n.Name()
	}
	if n, ok := node.(*Node); ok {
		if n.Name != "" {
			return n.Name
		}
		if n.Op != "" {
			return n.Op
		}
		return n.Kind.String()
	}
	return fmt.Sprintf("%T", node)
}

// EmitForward checks if node implements Emitter, calling its EmitForward. If node does not
// implement Emitter, it returns ErrEmissionUnsupported naming the node.
func EmitForward(node any, name string) (string, error) {
	em, ok := node.(Emitter)
	if !ok {
		return "", fmt.Errorf("%s: %w", NodeName(node), ErrEmissionUnsupported)
	}
	return em.EmitForward(name)
}

// EmitBackward checks if node implements BackwardEmitter, calling its EmitBackward. If node does
// not implement BackwardEmitter, it returns ErrBackwardEmissionUnsupported naming the node.
func EmitBackward(node any, name string, nThreads int) (string, error) {
	bem, ok := node.(BackwardEmitter)
	if !ok {
		return "", fmt.Errorf("%s: %w", NodeName(node), ErrBackwardEmissionUnsupported)
	}
	return bem.EmitBackward(name, nThreads)
}

// BackwardShmemBytes queries the dynamic shared memory needed for backward execution. If node does
// not implement BackwardEmitter, it returns ErrBackwardEmissionUnsupported.
func BackwardShmemBytes(node any, nThreads int, mode GradientMode) (int, error) {
	bem, ok := node.(BackwardEmitter)
	if !ok {
		return 0, fmt.Errorf("%s: %w", NodeName(node), ErrBackwardEmissionUnsupported)
	}
	return bem.BackwardShmemBytes(nThreads, mode), nil
}

func getTypes(n any) (inType, paramsType, computeType string) {
	inType = "float"
	paramsType = "float"
	computeType = "float"

	if te, ok := n.(TypedEmitter); ok {
		if it := te.InputType(); it != "" {
			inType = it
		}
		if pt := te.ParamsType(); pt != "" {
			paramsType = pt
		}
		if ct := te.ComputeType(); ct != "" {
			computeType = ct
		}
	}
	return inType, paramsType, computeType
}

// EmitFromBody generates the fixed forward CUDA __device__ function signature, emits lane-offset
// context preambles when enabled, and splices body. This matches tiny-cuda-nn object.h:665.
func EmitFromBody(n Emitter, name, body string) (string, error) {
	if n == nil {
		return "", fmt.Errorf("cudair: nil emitter")
	}
	if name == "" {
		return "", fmt.Errorf("cudair: empty function name")
	}
	inWidth := n.InputWidth()
	outWidth := n.OutputWidth()
	if inWidth <= 0 || outWidth <= 0 {
		return "", fmt.Errorf("cudair: invalid widths: in=%d out=%d", inWidth, outWidth)
	}

	ctxBytes, err := n.FwdCtxBytes()
	if err != nil {
		return "", fmt.Errorf("cudair: fwd ctx bytes: %w", err)
	}
	if ctxBytes < 0 {
		return "", fmt.Errorf("cudair: negative fwd ctx bytes: %d", ctxBytes)
	}

	inType, paramsType, computeType := getTypes(n)
	vecIn := fmt.Sprintf("tvec<%s, %d>", inType, inWidth)
	// Return type uses OutputWidth(), NOT PaddedOutputWidth() (object.h:662-663).
	vecOut := fmt.Sprintf("tvec<%s, %d>", computeType, outWidth)

	var offsetPreamble string
	if n.FwdCtxAlignedPerElement() && ctxBytes > 0 {
		offsetPreamble = fmt.Sprintf("\tif (fwd_ctx) fwd_ctx += lane_id() * %d;\n", ctxBytes)
	}
	offsetPreamble += forwardContextAccessor(ctxBytes)

	return fmt.Sprintf("__device__ auto %s(const %s& input, const %s* __restrict__ params, uint8_t* __restrict__ fwd_ctx = nullptr) -> %s {\n%s%s\n}",
		name, vecIn, paramsType, vecOut, offsetPreamble, body), nil
}

// EmitBackwardFromBody generates the fixed backward CUDA __device__ function signature, emits
// lane-offset context preambles when enabled, and splices body. This matches tiny-cuda-nn
// object.h:681.
func EmitBackwardFromBody(n BackwardEmitter, name, body string) (string, error) {
	if n == nil {
		return "", fmt.Errorf("cudair: nil backward emitter")
	}
	if name == "" {
		return "", fmt.Errorf("cudair: empty function name")
	}
	inWidth := n.InputWidth()
	outWidth := n.OutputWidth()
	if inWidth <= 0 || outWidth <= 0 {
		return "", fmt.Errorf("cudair: invalid widths: in=%d out=%d", inWidth, outWidth)
	}

	ctxBytes, err := n.FwdCtxBytes()
	if err != nil {
		return "", fmt.Errorf("cudair: fwd ctx bytes: %w", err)
	}
	if ctxBytes < 0 {
		return "", fmt.Errorf("cudair: negative fwd ctx bytes: %d", ctxBytes)
	}

	inType, paramsType, computeType := getTypes(n)
	vecIn := fmt.Sprintf("tvec<%s, %d>", inType, inWidth)
	vecOut := fmt.Sprintf("tvec<%s, %d>", computeType, outWidth)

	var offsetPreamble string
	if n.FwdCtxAlignedPerElement() && ctxBytes > 0 {
		offsetPreamble = fmt.Sprintf("\tfwd_ctx += lane_id() * %d;\n", ctxBytes)
	}
	offsetPreamble += backwardContextAccessor(ctxBytes)

	return fmt.Sprintf("__device__ void %s(const %s& dL_dy, const %s* __restrict__ params, const uint8_t* __restrict__ fwd_ctx, %s* __restrict__ dL_dparams = nullptr, %s* __restrict__ dL_dx = nullptr) {\n%s%s\n}",
		name, vecOut, paramsType, paramsType, vecIn, offsetPreamble, body), nil
}

func forwardContextAccessor(size int) string {
	if size == 0 {
		return ""
	}
	return fmt.Sprintf("\tstruct __cudair_context { uint8_t* p; int n; __device__ bool StoreByte(int i, uint8_t v) { if (i < 0 || i >= n || p == nullptr) return false; p[i] = v; return true; } __device__ bool LoadByte(int i, uint8_t* v) const { if (i < 0 || i >= n || p == nullptr || v == nullptr) return false; *v = p[i]; return true; } };\n\t__cudair_context __cudair_ctx{fwd_ctx, %d};\n", size)
}

func backwardContextAccessor(size int) string {
	if size == 0 {
		return ""
	}
	return fmt.Sprintf("\tstruct __cudair_context { const uint8_t* p; int n; __device__ bool LoadByte(int i, uint8_t* v) const { if (i < 0 || i >= n || p == nullptr || v == nullptr) return false; *v = p[i]; return true; } };\n\t__cudair_context __cudair_ctx{fwd_ctx, %d};\n", size)
}

// Unit carries concatenated device code, emitted symbol names, and resource requirements.
type Unit struct {
	Source      string
	Symbols     []string
	FwdCtxBytes int
	ShmemBytes  int
}

var identRegex = regexp.MustCompile(`[^a-zA-Z0-9_]`)

func sanitizeIdent(s string) string {
	clean := identRegex.ReplaceAllString(s, "_")
	clean = strings.Trim(clean, "_")
	if clean == "" {
		return "node"
	}
	return clean
}

func uniqueSymbol(base string, used map[string]bool) string {
	name := base
	for suffix := 2; used[name]; suffix++ {
		name = fmt.Sprintf("%s_%d", base, suffix)
	}
	used[name] = true
	return name
}

// EmitTree walks the emitter tree, verifies adjacent width agreement in Go, generates unique nested
// symbol names, and returns a concatenated Unit.
func EmitTree(root Emitter, name string) (Unit, error) {
	if root == nil {
		return Unit{}, fmt.Errorf("cudair: nil root emitter")
	}
	if name == "" {
		return Unit{}, fmt.Errorf("cudair: empty root name")
	}

	usedNames := map[string]bool{name: true}

	return emitTreeRecursive(root, name, usedNames)
}

// EmitBackwardTree emits backward device functions for a tree. Shared-memory demand is queried from
// each backward emitter for the requested launch shape.
func EmitBackwardTree(root BackwardEmitter, name string, nThreads int, mode GradientMode) (Unit, error) {
	if root == nil {
		return Unit{}, fmt.Errorf("cudair: nil backward root emitter")
	}
	if name == "" {
		return Unit{}, fmt.Errorf("cudair: empty root name")
	}
	if nThreads <= 0 {
		return Unit{}, fmt.Errorf("cudair: invalid backward thread count %d", nThreads)
	}
	usedNames := map[string]bool{name: true}
	return emitBackwardTreeRecursive(root, name, usedNames, nThreads, mode)
}

func emitBackwardTreeRecursive(node BackwardEmitter, name string, usedNames map[string]bool, nThreads int, mode GradientMode) (Unit, error) {
	var preambles []string
	var symbols []string
	var childSymbols []ChildSymbol
	ctxBytes := 0
	shmemBytes := node.BackwardShmemBytes(nThreads, mode)
	if shmemBytes < 0 {
		return Unit{}, fmt.Errorf("cudair: %s returned negative backward shared memory %d", NodeName(node), shmemBytes)
	}
	if comp, ok := node.(CompositeEmitter); ok {
		children := comp.Children()
		for i, child := range children {
			if child == nil {
				return Unit{}, fmt.Errorf("cudair: nil child at index %d under %s", i, name)
			}
			backward, ok := child.(BackwardEmitter)
			if !ok {
				return Unit{}, fmt.Errorf("%s: %w", NodeName(child), ErrBackwardEmissionUnsupported)
			}
			if i > 0 && children[i-1].OutputWidth() != child.InputWidth() {
				return Unit{}, fmt.Errorf("cudair: adjacent width mismatch between %s (output width %d) and %s (input width %d)", NodeName(children[i-1]), children[i-1].OutputWidth(), NodeName(child), child.InputWidth())
			}
			childName := uniqueSymbol(fmt.Sprintf("%s_%d_%s", name, i, sanitizeIdent(NodeName(child))), usedNames)
			sub, err := emitBackwardTreeRecursive(backward, childName, usedNames, nThreads, mode)
			if err != nil {
				return Unit{}, err
			}
			preambles = append(preambles, sub.Source)
			symbols = append(symbols, sub.Symbols...)
			childSymbols = append(childSymbols, ChildSymbol{Name: sub.Symbols[len(sub.Symbols)-1], FwdCtxOffset: ctxBytes})
			ctxBytes += sub.FwdCtxBytes
			if sub.ShmemBytes > shmemBytes {
				shmemBytes = sub.ShmemBytes
			}
		}
		if len(children) > 0 {
			if children[0].InputWidth() != node.InputWidth() {
				return Unit{}, fmt.Errorf("cudair: width mismatch between root %s (input width %d) and first child %s (input width %d)", NodeName(node), node.InputWidth(), NodeName(children[0]), children[0].InputWidth())
			}
			last := children[len(children)-1]
			if last.OutputWidth() != node.OutputWidth() {
				return Unit{}, fmt.Errorf("cudair: width mismatch between last child %s (output width %d) and root %s (output width %d)", NodeName(last), last.OutputWidth(), NodeName(node), node.OutputWidth())
			}
		}
	}
	declared, err := node.FwdCtxBytes()
	if err != nil {
		return Unit{}, err
	}
	ctxBytes += declared
	for i := range childSymbols {
		childSymbols[i].ParentFwdCtxBytes = declared
		childSymbols[i].ParentFwdCtxAligned = node.FwdCtxAlignedPerElement()
	}
	src := ""
	if composite, ok := node.(BackwardCompositeSymbolEmitter); ok && len(childSymbols) != 0 {
		src, err = composite.EmitBackwardWithChildren(name, childSymbols, nThreads, mode)
	} else {
		src, err = node.EmitBackward(name, nThreads)
	}
	if err != nil {
		return Unit{}, fmt.Errorf("cudair: %s emit backward: %w", NodeName(node), err)
	}
	var preamble string
	if node.FwdCtxAlignedPerElement() && declared > 0 {
		preamble = fmt.Sprintf("\tfwd_ctx += lane_id() * %d;\n", declared)
	}
	if err := validateEmittedContextUse(node, src, childSymbols, preamble); err != nil {
		return Unit{}, err
	}
	symbols = append(symbols, name)
	if len(preambles) != 0 {
		src = strings.Join(preambles, "\n\n") + "\n\n" + src
	}
	return Unit{Source: src, Symbols: symbols, FwdCtxBytes: ctxBytes, ShmemBytes: shmemBytes}, nil
}

func emitTreeRecursive(node Emitter, name string, usedNames map[string]bool) (Unit, error) {
	var preambles []string
	var symbols []string
	totalFwdCtx := 0
	maxShmem := 0
	var childSymbols []ChildSymbol

	// Check if compound node with children
	if comp, ok := node.(CompositeEmitter); ok {
		children := comp.Children()
		for i, child := range children {
			if child == nil {
				return Unit{}, fmt.Errorf("cudair: nil child at index %d under %s", i, name)
			}

			// Adjacent-width check between child nodes
			if i > 0 {
				prev := children[i-1]
				if prev.OutputWidth() != child.InputWidth() {
					return Unit{}, fmt.Errorf("cudair: adjacent width mismatch between %s (output width %d) and %s (input width %d)",
						NodeName(prev), prev.OutputWidth(), NodeName(child), child.InputWidth())
				}
			}

			childName := uniqueSymbol(fmt.Sprintf("%s_%d_%s", name, i, sanitizeIdent(NodeName(child))), usedNames)

			subUnit, err := emitTreeRecursive(child, childName, usedNames)
			if err != nil {
				return Unit{}, err
			}

			preambles = append(preambles, subUnit.Source)
			symbols = append(symbols, subUnit.Symbols...)
			childSymbol := subUnit.Symbols[len(subUnit.Symbols)-1]
			childSymbols = append(childSymbols, ChildSymbol{Name: childSymbol, FwdCtxOffset: totalFwdCtx})
			totalFwdCtx += subUnit.FwdCtxBytes
			if subUnit.ShmemBytes > maxShmem {
				maxShmem = subUnit.ShmemBytes
			}
		}

		// Boundary width check with parent root
		if len(children) > 0 {
			if children[0].InputWidth() != node.InputWidth() {
				return Unit{}, fmt.Errorf("cudair: width mismatch between root %s (input width %d) and first child %s (input width %d)",
					NodeName(node), node.InputWidth(), NodeName(children[0]), children[0].InputWidth())
			}
			lastChild := children[len(children)-1]
			if lastChild.OutputWidth() != node.OutputWidth() {
				return Unit{}, fmt.Errorf("cudair: width mismatch between last child %s (output width %d) and root %s (output width %d)",
					NodeName(lastChild), lastChild.OutputWidth(), NodeName(node), node.OutputWidth())
			}
		}
	}

	nodeFwdCtx, err := node.FwdCtxBytes()
	if err != nil {
		return Unit{}, fmt.Errorf("cudair: %s fwd ctx bytes: %w", NodeName(node), err)
	}
	totalFwdCtx += nodeFwdCtx
	for i := range childSymbols {
		childSymbols[i].ParentFwdCtxBytes = nodeFwdCtx
		childSymbols[i].ParentFwdCtxAligned = node.FwdCtxAlignedPerElement()
	}

	// Check shared memory requirement by asking the node
	if sp, ok := node.(ShmemProvider); ok {
		if req := sp.ShmemBytes(); req > maxShmem {
			maxShmem = req
		}
	}

	var src string
	if composite, ok := node.(CompositeSymbolEmitter); ok && len(childSymbols) != 0 {
		src, err = composite.EmitForwardWithChildren(name, childSymbols)
	} else {
		src, err = node.EmitForward(name)
	}
	if err != nil {
		return Unit{}, fmt.Errorf("cudair: %s emit forward: %w", NodeName(node), err)
	}
	var preamble string
	if node.FwdCtxAlignedPerElement() && nodeFwdCtx > 0 {
		preamble = fmt.Sprintf("\tif (fwd_ctx) fwd_ctx += lane_id() * %d;\n", nodeFwdCtx)
	}
	if err := validateEmittedContextUse(node, src, childSymbols, preamble); err != nil {
		return Unit{}, err
	}

	symbols = append(symbols, name)

	var fullSrc string
	if len(preambles) > 0 {
		fullSrc = strings.Join(preambles, "\n\n") + "\n\n" + src
	} else {
		fullSrc = src
	}

	return Unit{
		Source:      fullSrc,
		Symbols:     symbols,
		FwdCtxBytes: totalFwdCtx,
		ShmemBytes:  maxShmem,
	}, nil
}
