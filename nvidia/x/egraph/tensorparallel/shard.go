package tensorparallel

// ShardRows returns rows r of w, a row-major matrix of k columns: a
// column-parallel shard of a weight's output features.
func ShardRows[T any](w []T, k int, r Range) []T {
	return append([]T(nil), w[r.Lo*k:r.Hi*k]...)
}

// ShardCols returns columns r of every row of w, a row-major matrix of k
// columns: a row-parallel shard of a weight's input features.
func ShardCols[T any](w []T, k int, r Range) []T {
	n := len(w) / k
	out := make([]T, 0, n*r.Len())
	for i := range n {
		out = append(out, w[i*k+r.Lo:i*k+r.Hi]...)
	}
	return out
}

// QKV returns the rank's rows of the fused QKV weight [q; k; v], each
// [heads·head_dim, hidden], or of its bias with hidden 1.
func QKV[T any](rp RankPlan, wq, wk, wv []T, hidden int) []T {
	hd := rp.headDim
	q := ShardRows(wq, hidden, Range{rp.QHeads.Lo * hd, rp.QHeads.Hi * hd})
	k := ShardRows(wk, hidden, Range{rp.KVHeads.Lo * hd, rp.KVHeads.Hi * hd})
	v := ShardRows(wv, hidden, Range{rp.KVHeads.Lo * hd, rp.KVHeads.Hi * hd})
	return append(append(q, k...), v...)
}

// GateUp returns the rank's rows of the fused [gate; up] weight, each
// [intermediate, hidden].
func GateUp[T any](rp RankPlan, wg, wu []T) []T {
	return append(ShardRows(wg, rp.hidden, rp.Inter), ShardRows(wu, rp.hidden, rp.Inter)...)
}

// O returns the rank's columns of w_o [hidden, heads·head_dim].
func O[T any](rp RankPlan, wo []T) []T {
	hd := rp.headDim
	return ShardCols(wo, len(wo)/rp.hidden, Range{rp.QHeads.Lo * hd, rp.QHeads.Hi * hd})
}

// Down returns the rank's columns of w_down [hidden, intermediate].
func Down[T any](rp RankPlan, wd []T) []T {
	return ShardCols(wd, rp.inter, rp.Inter)
}
