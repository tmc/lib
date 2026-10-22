package dsv4vision

import "fmt"

// nextImage parses the pinned synthetic-ID grammar. Leading pads belong to
// the embedding block but not the bidirectional interval.
func nextImage(tokens []int32, vocab int32, cursor int) (block, start, end, next int, found bool, err error) {
	i := cursor
	for i < len(tokens) && tokens[i] >= 0 && tokens[i] < vocab {
		i++
	}
	if i == len(tokens) {
		return 0, 0, 0, i, false, nil
	}
	if tokens[i] < 0 {
		return 0, 0, 0, i, false, fmt.Errorf("negative image token")
	}
	block = i
	for i < len(tokens) && tokens[i] == vocab+int32(Pad) {
		i++
	}
	if i == len(tokens) || tokens[i] != vocab+int32(Start) {
		return 0, 0, 0, i, false, fmt.Errorf("missing image-start marker")
	}
	start = i
	i++
	for i < len(tokens) {
		t := tokens[i]
		if t < vocab || t > vocab+int32(End) || t == vocab+int32(Start) {
			return 0, 0, 0, i, false, fmt.Errorf("invalid token within image")
		}
		if t == vocab+int32(End) {
			return block, start, i, i + 1, true, nil
		}
		i++
	}
	return 0, 0, 0, i, false, fmt.Errorf("missing image-end marker")
}

// AttentionBounds returns inclusive raw-key position bounds for each query.
// tokens must contain complete image blocks. pos0 is the first query position,
// nRaw is the retained raw-key count, and window zero means unlimited history.
// Compressed-key visibility and indexer masks are separate backend contracts.
func AttentionBounds(tokens []int32, vocab, pos0, nRaw, window int) ([][2]uint32, error) {
	if len(tokens) == 0 || vocab < 1 || vocab > 1<<31-5 || pos0 < 0 || uint64(pos0)+uint64(len(tokens)) > 1<<32-1 || nRaw < 1 || uint64(nRaw) > uint64(pos0)+uint64(len(tokens)) || window < 0 || uint64(window) > 1<<32-1 {
		return nil, fmt.Errorf("invalid visual attention dimensions")
	}
	first, last := pos0+len(tokens)-nRaw, pos0+len(tokens)-1
	b := make([][2]uint32, len(tokens))
	for q := range tokens {
		qpos := pos0 + q
		lo := first
		if window != 0 && qpos+1 > window {
			lo = max(lo, qpos+1-window)
		}
		b[q] = [2]uint32{uint32(lo), uint32(min(qpos, last))}
	}
	cursor := 0
	for {
		_, start, end, next, found, err := nextImage(tokens, int32(vocab), cursor)
		if err != nil {
			return nil, err
		}
		if !found {
			break
		}
		cursor = next
		for q := start; q <= end; q++ {
			qpos := pos0 + q
			left, right := min(q-start, 383), min(end-q, 384)
			lo := 0
			if window != 0 {
				add := 0
				if left >= window {
					add = left - (window - 1)
				}
				lo = max(0, qpos-(window-1+add))
			}
			b[q] = [2]uint32{uint32(max(lo, first)), uint32(min(qpos+right, last))}
		}
	}
	return b, nil
}

// PrefillChunk returns a nonempty chunk ending outside an image block. cap
// limits the chunk size. It refuses a block larger than cap at the start.
func PrefillChunk(tokens []int32, vocab, pos0, cap int) (int, error) {
	if vocab < 1 || vocab > 1<<31-5 || cap < 1 || pos0 < 0 || pos0 >= len(tokens) {
		return 0, fmt.Errorf("invalid visual prefill chunk")
	}
	limit := pos0 + min(len(tokens)-pos0, cap)
	cursor := pos0
	for {
		block, _, end, next, found, err := nextImage(tokens, int32(vocab), cursor)
		if err != nil {
			return 0, err
		}
		if !found || block >= limit {
			break
		}
		cursor = next
		after := end + 1
		if after > limit {
			if after-pos0 > cap {
				if block == pos0 {
					return 0, fmt.Errorf("image block exceeds prefill chunk")
				}
				limit = block
				break
			}
			limit = after
		}
	}
	if limit <= pos0 {
		return 0, fmt.Errorf("empty visual prefill chunk")
	}
	return limit - pos0, nil
}
