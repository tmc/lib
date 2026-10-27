package hopper

import "fmt"

// WorkspaceFormula is a closed-form workspace size in bytes that a host
// launcher can evaluate without calling Go: with s = ceil(maxKeys/Chunk)
// splits and g = ceil(s/Group) merge groups,
//
//	Bytes(batch, maxKeys) = batch * (A + B*s + D*g)
type WorkspaceFormula struct {
	Chunk int // keys per split
	Group int // splits per merge group
	A     int // constant bytes per batch element
	B     int // bytes per split per batch element
	D     int // bytes per merge group per batch element
}

// Bytes returns the workspace size in bytes for batch requests of at most
// maxKeys keys according to the fitted formula.
func (w WorkspaceFormula) Bytes(batch, maxKeys int) int {
	if batch <= 0 || maxKeys <= 0 || w.Chunk <= 0 || w.Group <= 0 {
		return 0
	}
	s := (maxKeys + w.Chunk - 1) / w.Chunk
	g := (s + w.Group - 1) / w.Group
	return batch * (w.A + w.B*s + w.D*g)
}

// FitWorkspaceFormula fits the affine split/merge formula
//
//	batch * (a + b*splits + d*ceil(splits/group))
//
// to workspaceBytes(batch, maxKeys) and returns an error if the function
// cannot be represented in that form or disagrees at any verification point.
func FitWorkspaceFormula(workspaceBytes func(batch, maxKeys int) int) (WorkspaceFormula, error) {
	var w WorkspaceFormula
	if workspaceBytes == nil {
		return w, fmt.Errorf("fit workspace formula: nil function")
	}
	base := workspaceBytes(1, 1)
	if base <= 0 {
		return w, fmt.Errorf("fit workspace formula: workspaceBytes(1, 1) = %d, want > 0", base)
	}
	const maxChunkProbe = 65536
	for k := 2; k <= maxChunkProbe+1; k++ {
		if workspaceBytes(1, k) != base {
			w.Chunk = k - 1
			break
		}
	}
	if w.Chunk == 0 {
		return w, fmt.Errorf("fit workspace formula: workspace size did not vary with maxKeys up to %d", maxChunkProbe)
	}
	at := func(s int) int { return workspaceBytes(1, s*w.Chunk) }
	if got, want := workspaceBytes(1, w.Chunk+1), at(2); got != want {
		return w, fmt.Errorf("fit workspace formula: chunk boundary %d: workspaceBytes(1, %d) = %d != %d", w.Chunk, w.Chunk+1, got, want)
	}
	w.B = at(2) - at(1)
	for s := 3; s <= 1024 && w.Group == 0; s++ {
		if at(s)-at(s-1) != w.B {
			w.Group = s - 1
		}
	}
	if w.Group == 0 {
		w.Group = 1 << 30 // one group at every size checked
	} else {
		w.D = at(w.Group+1) - at(w.Group) - w.B
	}
	w.A = at(1) - w.B - w.D

	points := [][2]int{
		{1, 1},
		{1, w.Chunk},
		{1, w.Chunk + 1},
		{2, 2*w.Chunk + 1},
		{3, 4097},
		{8, 33000},
		{2, 131072},
		{5, 4096*17 + 1},
	}
	if w.Group < 1<<20 {
		points = append(points,
			[2]int{1, w.Group * w.Chunk},
			[2]int{1, w.Group*w.Chunk + 1},
			[2]int{4, 2*w.Group*w.Chunk + 1},
		)
	}
	for _, bk := range points {
		if got, want := w.Bytes(bk[0], bk[1]), workspaceBytes(bk[0], bk[1]); got != want {
			return w, fmt.Errorf("fit workspace formula: model %d != %d at batch=%d maxKeys=%d", got, want, bk[0], bk[1])
		}
	}
	return w, nil
}

// WorkspaceFormula fits a's WorkspaceBytes to a closed-form WorkspaceFormula
// and verifies the fit across representative (batch, maxKeys) points.
func (a PagedAttention) WorkspaceFormula() (WorkspaceFormula, error) {
	normed, _, err := a.norm()
	if err != nil {
		return WorkspaceFormula{}, err
	}
	if normed.Splits(normed.Chunk) != 1 || normed.Splits(normed.Chunk+1) != 2 {
		return WorkspaceFormula{}, fmt.Errorf("paged attention %s: chunk is not %d", normed.Name, normed.Chunk)
	}
	w, err := FitWorkspaceFormula(normed.WorkspaceBytes)
	if err != nil {
		return w, fmt.Errorf("paged attention %s: %w", normed.Name, err)
	}
	if w.Chunk != normed.Chunk {
		return w, fmt.Errorf("paged attention %s: fitted chunk %d != %d", normed.Name, w.Chunk, normed.Chunk)
	}
	return w, nil
}
