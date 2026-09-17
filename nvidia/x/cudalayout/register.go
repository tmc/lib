package cudalayout

// RegisterOwner identifies the lane and register slot owning an element.
// Lane is in [0,32); Register is a logical slot, independent of pressure.
type RegisterOwner struct{ Lane, Register int }

// Owner maps a tile element to a deterministic 32-lane register tile. The
// mapping uses linear element order and therefore also handles tails.
func (t Tile) Owner(row, col int) (RegisterOwner, error) {
	p, err := t.Index(row, col)
	if err != nil {
		return RegisterOwner{}, err
	}
	return RegisterOwner{Lane: p.ElementOffset % 32, Register: p.ElementOffset / 32}, nil
}

// OwnerFor returns the logical element represented by an owner slot.
func (t Tile) OwnerFor(lane, register int) (row, col int, err error) {
	if t.rows == 0 || t.cols == 0 || t.stride == 0 {
		return 0, 0, ErrUnsupported
	}
	if lane < 0 || lane >= 32 || register < 0 {
		return 0, 0, ErrOutOfBounds
	}
	if register > (int(^uint(0)>>1)-lane)/32 {
		return 0, 0, ErrOverflow
	}
	e := register*32 + lane
	if register > (t.rows*t.stride)/32 || e < 0 {
		return 0, 0, ErrOutOfBounds
	}
	row, col = e/t.stride, e%t.stride
	if row >= t.rows || col >= t.cols {
		return 0, 0, ErrOutOfBounds
	}
	return row, col, nil
}

// RedistributionPermutation describes a transpose-like tile redistribution.
type RedistributionPermutation struct{ Rows, Cols int }

// TransposePermutation returns the independent logical transpose mapping.
func TransposePermutation(rows, cols int) (RedistributionPermutation, error) {
	if rows <= 0 || cols <= 0 {
		return RedistributionPermutation{}, ErrOutOfBounds
	}
	return RedistributionPermutation{Rows: rows, Cols: cols}, nil
}
func (p RedistributionPermutation) Map(row, col int) (int, int, error) {
	if row < 0 || row >= p.Rows || col < 0 || col >= p.Cols {
		return 0, 0, ErrOutOfBounds
	}
	return col, row, nil
}

// TransposeOwners maps a source element through a logical transpose and
// returns the destination register owner.
func (t Tile) TransposeOwners(row, col int) (RegisterOwner, error) {
	if row < 0 || row >= t.rows || col < 0 || col >= t.cols {
		return RegisterOwner{}, ErrOutOfBounds
	}
	dst, err := NewTile(t.cols, t.rows, t.elemBytes)
	if err != nil {
		return RegisterOwner{}, err
	}
	return dst.Owner(col, row)
}
