package hftokenizer

import "container/heap"

type pair [2]int32
type merge struct {
	rank int
	id   int32
}
type piece struct {
	id         int32
	prev, next int
}
type candidate struct {
	left, right int
	ids         pair
	merge       merge
}
type mergeHeap []candidate

func (h mergeHeap) Len() int { return len(h) }
func (h mergeHeap) Less(i, j int) bool {
	return h[i].merge.rank < h[j].merge.rank || h[i].merge.rank == h[j].merge.rank && h[i].left < h[j].left
}
func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *mergeHeap) Push(v any)   { *h = append(*h, v.(candidate)) }
func (h *mergeHeap) Pop() any {
	old := *h
	v := old[len(old)-1]
	*h = old[:len(old)-1]
	return v
}

func (t *Tokenizer) encodePiece(s string, ids []int32) []int32 {
	if len(s) == 0 {
		return ids
	}
	if len(s) == 1 {
		return append(ids, t.bytes[s[0]])
	}
	nodes := make([]piece, len(s))
	for i := range nodes {
		nodes[i] = piece{id: t.bytes[s[i]], prev: i - 1, next: i + 1}
	}
	nodes[len(nodes)-1].next = -1
	var queue mergeHeap
	add := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		ids := pair{nodes[left].id, nodes[right].id}
		if m, ok := t.merges[ids]; ok {
			heap.Push(&queue, candidate{left: left, right: right, ids: ids, merge: m})
		}
	}
	for i := 0; i < len(nodes)-1; i++ {
		add(i, i+1)
	}
	for len(queue) > 0 {
		c := heap.Pop(&queue).(candidate)
		left, right := &nodes[c.left], &nodes[c.right]
		if left.next != c.right || left.id != c.ids[0] || right.id != c.ids[1] {
			continue
		}
		left.id = c.merge.id
		left.next = right.next
		right.id = -1
		if left.next >= 0 {
			nodes[left.next].prev = c.left
		}
		add(left.prev, c.left)
		add(c.left, left.next)
	}
	for i := 0; i >= 0; i = nodes[i].next {
		ids = append(ids, nodes[i].id)
	}
	return ids
}
