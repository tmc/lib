package egraph

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudashape"
)

var (
	// ErrIncompatibleShape is returned when rewriting combines incompatible tensor shapes.
	ErrIncompatibleShape = errors.New("egraph: incompatible shapes in rewrite")

	// ErrIncompatibleDType is returned when rewriting combines incompatible data types.
	ErrIncompatibleDType = errors.New("egraph: incompatible data types in rewrite")

	// ErrIncompatibleLayout is returned when rewriting combines incompatible memory layouts.
	ErrIncompatibleLayout = errors.New("egraph: incompatible layouts in rewrite")

	// ErrEffectBoundaryViolation is returned when rewriting violates memory effect boundaries.
	ErrEffectBoundaryViolation = errors.New("egraph: effect boundary violated in rewrite")

	// ErrGroupingDependenceViolation is returned when a cross-token grouping rewrite violates causal/data dependence.
	ErrGroupingDependenceViolation = errors.New("egraph: grouping rewrite violates dependence precondition")

	// ErrUnalignedVectorAccess is returned when a vector load/store violates alignment preconditions.
	ErrUnalignedVectorAccess = errors.New("egraph: unaligned vector memory access in rewrite")
	ErrMissingTypedMetadata  = errors.New("egraph: mandatory rewrite requires typed metadata")
	ErrInvalidRewriteProof   = errors.New("egraph: rewrite proof precondition not met")
)

// TypedMetadata holds typed contract information for an e-class or expression node.
type TypedMetadata struct {
	Shape     cudashape.Shape
	DType     cudair.DataType
	Layout    cudashape.Layout
	Effect    cudair.StorageEffect
	Alignment int // in bytes (e.g. 16 for 128-bit vector loads)
	// LoopInvariant and NumericalPolicy are proof facts supplied by the
	// frontend.  A zero value is deliberately not an assertion of safety.
	LoopInvariant   bool
	NumericalPolicy string
}

// Clone returns a copy of TypedMetadata.
func (m TypedMetadata) Clone() TypedMetadata {
	return m
}

// CheckLegality verifies whether combining or substituting two typed contracts is legal.
func CheckLegality(current, proposed TypedMetadata) error {
	// 1. Storage Effect boundary check
	// Mutating operations cannot be silently merged with ReadOnly views or cross effect boundaries.
	if current.Effect != proposed.Effect {
		return fmt.Errorf("%w: cannot merge storage effects (%v vs %v)",
			ErrEffectBoundaryViolation, current.Effect, proposed.Effect)
	}

	// 2. Data type check
	if current.DType != proposed.DType {
		return fmt.Errorf("%w: current %v, proposed %v", ErrIncompatibleDType, current.DType, proposed.DType)
	}

	// 3. Shape check (if non-scalar)
	if current.Shape.Rank() > 0 || proposed.Shape.Rank() > 0 {
		if !current.Shape.Equal(proposed.Shape) {
			return fmt.Errorf("%w: current %v, proposed %v", ErrIncompatibleShape, current.Shape, proposed.Shape)
		}
	}

	// 4. Layout check
	if current.Layout != proposed.Layout {
		return fmt.Errorf("%w: current %v, proposed %v", ErrIncompatibleLayout, current.Layout, proposed.Layout)
	}

	// 5. Alignment is part of the representation contract. Rewrites may not
	// silently change it; callers that need a different vector width must first
	// insert an explicitly checked conversion.
	if proposed.Alignment > 0 && current.Alignment > 0 {
		if current.Alignment != proposed.Alignment {
			return fmt.Errorf("%w: current alignment %d bytes does not satisfy required %d bytes",
				ErrUnalignedVectorAccess, current.Alignment, proposed.Alignment)
		}
	}

	return nil
}

// LegalAdapter wraps an EGraph with typed metadata tracking and legality gates.
type LegalAdapter struct {
	g        *EGraph
	metadata map[Id]TypedMetadata
}

// NewLegalAdapter creates a LegalAdapter for the given EGraph.
func NewLegalAdapter(g *EGraph) *LegalAdapter {
	return &LegalAdapter{
		g:        g,
		metadata: make(map[Id]TypedMetadata),
	}
}

// SetMetadata records typed metadata for an e-class.
func (a *LegalAdapter) SetMetadata(id Id, meta TypedMetadata) {
	a.metadata[a.g.Find(id)] = meta
}

// GetMetadata retrieves typed metadata for an e-class.
func (a *LegalAdapter) GetMetadata(id Id) (TypedMetadata, bool) {
	meta, ok := a.metadata[a.g.Find(id)]
	return meta, ok
}

// LegalUnion merges two equivalence classes only after validating typed legality.
func (a *LegalAdapter) LegalUnion(id1, id2 Id) (Id, bool, error) {
	root1 := a.g.Find(id1)
	root2 := a.g.Find(id2)
	if root1 == root2 {
		return root1, false, nil
	}

	meta1, ok1 := a.metadata[root1]
	meta2, ok2 := a.metadata[root2]

	if ok1 && ok2 {
		if err := CheckLegality(meta1, meta2); err != nil {
			return root1, false, err
		}
	}

	newRoot, merged := a.g.Union(root1, root2)
	if merged {
		// Retain metadata on new root
		if ok1 {
			a.metadata[newRoot] = meta1
		} else if ok2 {
			a.metadata[newRoot] = meta2
		}
		if newRoot != root1 {
			delete(a.metadata, root1)
		}
		if newRoot != root2 {
			delete(a.metadata, root2)
		}
	}
	return newRoot, merged, nil
}
