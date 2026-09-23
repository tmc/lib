//go:build !darwin

package main

import "errors"

func (*productionPrefixLease) Restore([][]float32, [][]float32, int) error {
	return errors.New("observed output: prefix restore unsupported")
}

func (*ModelEngine) restoreScheduledPrefix([]int32, *SequenceKVCache, Config) (int, int32) {
	return 0, 0
}

func (*ModelEngine) storeScheduledPrefix([]int32, *SequenceKVCache, int32, Config) {}

func (*ModelEngine) restoreScheduledPrefixRow([]int32, *BatchedAttentionCache, int) (int, int32) {
	return 0, 0
}

func (*ModelEngine) storeScheduledPrefixRow([]int32, *BatchedAttentionCache, int, int32) {}
