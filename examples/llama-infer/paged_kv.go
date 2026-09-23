package main

// Host-side half of the paged KV path: the switches that select it and the
// side tables one paged step uploads. Nothing here touches a device, so this
// file has no build tag and its tests run on every host. The device half is
// paged_kv_cuda.go and paged_kv_step_cuda.go.

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

// pagedKVEnabled reports whether the paged KV path was explicitly requested.
// It is opt-in rather than a Config field so that enabling it is always a
// deliberate act by whoever is holding the device.
func pagedKVEnabled() bool { return os.Getenv("LLAMA_INFER_PAGED_KV") == "1" }

// defaultPagedKVSequences is how many serving sequences the block pool holds
// room for when LLAMA_INFER_PAGED_KV_SEQS is unset.
const defaultPagedKVSequences = 8

// pagedKVSequences reads LLAMA_INFER_PAGED_KV_SEQS, the number of serving
// sequences the block pool is sized for. The pool is fixed when the engine is
// compiled, before any server names its capacity, so this must be at least the
// serving capacity or admission fails with an out-of-blocks error.
func pagedKVSequences() (int, error) {
	s := os.Getenv("LLAMA_INFER_PAGED_KV_SEQS")
	if s == "" {
		return defaultPagedKVSequences, nil
	}
	n, err := strconv.Atoi(s)
	if err != nil || n < 1 {
		return 0, fmt.Errorf("LLAMA_INFER_PAGED_KV_SEQS=%q: want a positive integer", s)
	}
	return n, nil
}

// pagedKVConfig sizes the block pool. TotalBlocks is the pool shared by every
// resident sequence, which is the whole point: MaxBlocksPerSeq bounds one
// sequence's block table but reserves nothing.
type pagedKVConfig struct {
	BlockSize       int
	TotalBlocks     int
	MaxBlocksPerSeq int
}

// defaultPagedKVConfig derives a pool from the model's context window and the
// number of sequences the pool must hold at once. BlockSize 16 matches the
// Metal prefix cache's block size, so the two backends describe prefixes the
// same way.
func defaultPagedKVConfig(maxSeqLen, capacity int) pagedKVConfig {
	const blockSize = 16
	perSeq := (maxSeqLen + blockSize - 1) / blockSize
	if capacity < 1 {
		capacity = 1
	}
	return pagedKVConfig{
		BlockSize:       blockSize,
		TotalBlocks:     perSeq * capacity,
		MaxBlocksPerSeq: perSeq,
	}
}

// pagedKVTables are the int32 side tables the paged kernels read for one
// step. Every row of the step is its own sequence entry: seqIndex[t] is t,
// positions[t] is the row's position, seqLens[t] is positions[t]+1, and
// blockTable is row-major [rows][maxBlocksPerSeq]. Two rows of one prefill tile
// therefore carry the same block table twice, and a row attends over exactly
// the positions at or before its own, which is the dense batch path's causal
// rule.
type pagedKVTables struct {
	seqIndex, positions, seqLens, blockTable []int32
}

// buildPagedKVTables lays out the tables for rows whose block tables and
// positions are given. Each table must cover its row's position. Unused table
// entries are -1: both paged kernels report a negative slot through their
// status word, so a read past a table's end is caught rather than served from
// whatever slot happened to sit there.
func buildPagedKVTables(tables [][]int, positions []int, blockSize, maxBlocksPerSeq int) (pagedKVTables, error) {
	var t pagedKVTables
	if len(tables) == 0 {
		return t, errors.New("paged kv: no rows")
	}
	if len(tables) != len(positions) {
		return t, fmt.Errorf("paged kv: %d block tables for %d positions", len(tables), len(positions))
	}
	if blockSize < 1 || maxBlocksPerSeq < 1 {
		return t, fmt.Errorf("paged kv: block size %d and table width %d must be positive", blockSize, maxBlocksPerSeq)
	}
	rows := len(tables)
	t.seqIndex = make([]int32, rows)
	t.positions = make([]int32, rows)
	t.seqLens = make([]int32, rows)
	t.blockTable = make([]int32, rows*maxBlocksPerSeq)
	for i := range t.blockTable {
		t.blockTable[i] = -1
	}
	for row, table := range tables {
		pos := positions[row]
		if pos < 0 {
			return pagedKVTables{}, fmt.Errorf("paged kv: row %d at negative position %d", row, pos)
		}
		if len(table) > maxBlocksPerSeq {
			return pagedKVTables{}, fmt.Errorf("paged kv: row %d has %d blocks, table width is %d", row, len(table), maxBlocksPerSeq)
		}
		if len(table)*blockSize <= pos {
			return pagedKVTables{}, fmt.Errorf("paged kv: row %d has %d blocks of %d tokens, which do not cover position %d", row, len(table), blockSize, pos)
		}
		t.seqIndex[row] = int32(row)
		t.positions[row] = int32(pos)
		t.seqLens[row] = int32(pos + 1)
		for i, slot := range table {
			t.blockTable[row*maxBlocksPerSeq+i] = int32(slot)
		}
	}
	return t, nil
}
