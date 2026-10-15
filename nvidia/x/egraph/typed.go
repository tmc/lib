package egraph

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/tmc/lib/egraphcore"
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
	ErrUnalignedVectorAccess       = errors.New("egraph: unaligned vector memory access in rewrite")
	ErrIncompatibleNumericalPolicy = errors.New("egraph: incompatible numerical policies in rewrite")
	ErrMissingTypedMetadata        = errors.New("egraph: mandatory rewrite requires typed metadata")
	ErrInvalidRewriteProof         = errors.New("egraph: rewrite proof precondition not met")
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
	if current.Alignment != proposed.Alignment {
		return fmt.Errorf("%w: current alignment %d bytes does not satisfy required %d bytes",
			ErrUnalignedVectorAccess, current.Alignment, proposed.Alignment)
	}
	if current.NumericalPolicy != proposed.NumericalPolicy {
		return fmt.Errorf("%w: current %q, proposed %q", ErrIncompatibleNumericalPolicy, current.NumericalPolicy, proposed.NumericalPolicy)
	}

	return nil
}

// mergeMetadata combines matching contracts without upgrading unknown facts.
// LoopInvariant is true only when both inputs assert it.
func mergeMetadata(current, proposed TypedMetadata) (TypedMetadata, error) {
	if err := CheckLegality(current, proposed); err != nil {
		return TypedMetadata{}, err
	}
	merged := current.Clone()
	merged.LoopInvariant = current.LoopInvariant && proposed.LoopInvariant
	return merged, nil
}

// LegalAdapter wraps an EGraph with typed metadata tracking and legality gates.
type LegalAdapter struct {
	g           *EGraph
	metadataErr error
}

// NewLegalAdapter creates a LegalAdapter for the given EGraph.
func NewLegalAdapter(g *EGraph) *LegalAdapter {
	a := &LegalAdapter{g: g}
	if g == nil {
		a.metadataErr = fmt.Errorf("egraph: nil graph")
		return a
	}
	_, a.metadataErr = g.Facts().Register(typedMetadataSchema())
	return a
}

func typedMetadataSchema() egraphcore.Schema {
	return egraphcore.Schema{
		Name: "nvidia.typed_metadata",
		Columns: []egraphcore.Column{
			{Kind: egraphcore.ClassValue}, {Kind: egraphcore.StringValue},
			{Kind: egraphcore.Int64Value}, {Kind: egraphcore.Int64Value},
			{Kind: egraphcore.Int64Value}, {Kind: egraphcore.Int64Value},
			{Kind: egraphcore.BoolValue}, {Kind: egraphcore.StringValue},
		},
	}
}

func (a *LegalAdapter) metadataRelation() (*egraphcore.Relation[Id], error) {
	if a == nil || a.g == nil {
		return nil, fmt.Errorf("egraph: nil legal adapter")
	}
	if a.metadataErr != nil {
		return nil, a.metadataErr
	}
	relation := a.g.Facts().Get("nvidia.typed_metadata")
	if relation == nil {
		return nil, fmt.Errorf("egraph: typed metadata relation is not registered")
	}
	return relation, nil
}

// SetMetadata records typed metadata for an e-class.
func (a *LegalAdapter) SetMetadata(id Id, meta TypedMetadata) error {
	relation, err := a.metadataRelation()
	if err != nil {
		return err
	}
	root := a.g.Find(id)
	staged := a.g.Facts().Clone()
	relation = staged.Get("nvidia.typed_metadata")
	rows, complete, err := relation.Query([]egraphcore.Filter[Id]{{Column: 0, Value: egraphcore.Class(root)}}, a.g.Find, nil)
	if err != nil {
		return fmt.Errorf("egraph: query typed metadata for update: %w", err)
	}
	if !complete {
		return fmt.Errorf("egraph: typed metadata update query was incomplete")
	}
	for _, row := range rows {
		if _, err := relation.Delete(row, a.g.Find); err != nil {
			return fmt.Errorf("egraph: delete typed metadata: %w", err)
		}
	}
	if _, err := relation.Add(metadataValues(root, meta), a.g.Find); err != nil {
		return fmt.Errorf("egraph: add typed metadata: %w", err)
	}
	if err := a.g.Facts().ReplaceFrom(staged, nil); err != nil {
		return fmt.Errorf("egraph: commit typed metadata: %w", err)
	}
	return nil
}

// GetMetadata retrieves typed metadata for an e-class. It returns false when
// metadata is missing or cannot be read; use GetMetadataChecked to distinguish
// those cases.
func (a *LegalAdapter) GetMetadata(id Id) (TypedMetadata, bool) {
	meta, ok, _ := a.getMetadata(id)
	return meta, ok
}

// GetMetadataChecked retrieves typed metadata and reports query or decoding
// errors separately from absent metadata.
func (a *LegalAdapter) GetMetadataChecked(id Id) (TypedMetadata, bool, error) {
	return a.getMetadata(id)
}

func (a *LegalAdapter) getMetadata(id Id) (TypedMetadata, bool, error) {
	relation, err := a.metadataRelation()
	if err != nil {
		return TypedMetadata{}, false, err
	}
	rows, complete, err := relation.Query([]egraphcore.Filter[Id]{{Column: 0, Value: egraphcore.Class(a.g.Find(id))}}, a.g.Find, nil)
	if err != nil {
		return TypedMetadata{}, false, fmt.Errorf("egraph: query typed metadata: %w", err)
	}
	if !complete {
		return TypedMetadata{}, false, fmt.Errorf("egraph: typed metadata query was incomplete")
	}
	if len(rows) == 0 {
		return TypedMetadata{}, false, nil
	}
	if len(rows) != 1 {
		return TypedMetadata{}, false, fmt.Errorf("egraph: multiple typed metadata facts for class %d", a.g.Find(id))
	}
	meta, err := decodeMetadata(rows[0])
	if err != nil {
		return TypedMetadata{}, false, fmt.Errorf("egraph: decode typed metadata: %w", err)
	}
	return meta, true, nil
}

// LegalUnion merges two readonly equivalence classes with matching metadata.
// Both classes must have metadata. Unknown alignment or numerical policy may
// match unknown facts, but cannot acquire a known fact from the other class.
// Alias and mutation effects require storage identity and ordering proofs not
// represented by TypedMetadata, so distinct effectful classes are refused.
func (a *LegalAdapter) LegalUnion(id1, id2 Id) (Id, bool, error) {
	if _, err := a.metadataRelation(); err != nil {
		return 0, false, err
	}
	root1 := a.g.Find(id1)
	root2 := a.g.Find(id2)
	if root1 == root2 {
		return root1, false, nil
	}

	meta1, ok1, err := a.getMetadata(root1)
	if err != nil {
		return root1, false, err
	}
	meta2, ok2, err := a.getMetadata(root2)
	if err != nil {
		return root1, false, err
	}

	if !ok1 || !ok2 {
		return root1, false, ErrMissingTypedMetadata
	}
	if meta1.Effect != cudair.EffectReadOnly || meta2.Effect != cudair.EffectReadOnly {
		return root1, false, ErrEffectBoundaryViolation
	}
	mergedMetadata, err := mergeMetadata(meta1, meta2)
	if err != nil {
		return root1, false, err
	}

	trial := a.g.Clone()
	newRoot, merged, err := trial.UnionChecked(root1, root2)
	if err != nil {
		return newRoot, merged, fmt.Errorf("egraph: union typed classes: %w", err)
	}
	if merged {
		trialAdapter := &LegalAdapter{g: trial}
		relation, err := trialAdapter.metadataRelation()
		if err != nil {
			return newRoot, merged, err
		}
		if err := relation.Recanonicalize(trial.Find); err != nil {
			return newRoot, merged, fmt.Errorf("egraph: recanonicalize typed metadata: %w", err)
		}
		if ok1 || ok2 {
			if err := trialAdapter.SetMetadata(newRoot, mergedMetadata); err != nil {
				return newRoot, merged, err
			}
		}
	}
	*a.g = *trial
	return newRoot, merged, nil
}

func metadataValues(id Id, meta TypedMetadata) []egraphcore.Value[Id] {
	return []egraphcore.Value[Id]{
		egraphcore.Class[Id](id), egraphcore.String[Id](meta.Shape.String()),
		egraphcore.Int64[Id](int64(meta.DType)), egraphcore.Int64[Id](int64(meta.Layout)),
		egraphcore.Int64[Id](int64(meta.Effect)), egraphcore.Int64[Id](int64(meta.Alignment)),
		egraphcore.Bool[Id](meta.LoopInvariant), egraphcore.String[Id](meta.NumericalPolicy),
	}
}

func decodeMetadata(row []egraphcore.Value[Id]) (TypedMetadata, error) {
	if len(row) != 8 {
		return TypedMetadata{}, fmt.Errorf("egraph: invalid typed metadata row")
	}
	shapeText := strings.Trim(row[1].String, "[]")
	var dims []int
	if shapeText != "" {
		for _, part := range strings.Split(shapeText, ",") {
			dim, err := strconv.Atoi(strings.TrimSpace(part))
			if err != nil {
				return TypedMetadata{}, fmt.Errorf("egraph: invalid typed metadata shape: %w", err)
			}
			dims = append(dims, dim)
		}
	}
	shape, err := cudashape.New(dims...)
	if err != nil {
		return TypedMetadata{}, fmt.Errorf("egraph: invalid typed metadata shape: %w", err)
	}
	return TypedMetadata{
		Shape: shape, DType: cudair.DataType(row[2].Int64), Layout: cudashape.Layout(row[3].Int64),
		Effect: cudair.StorageEffect(row[4].Int64), Alignment: int(row[5].Int64),
		LoopInvariant: row[6].Bool, NumericalPolicy: row[7].String,
	}, nil
}
