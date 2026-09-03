//go:build !darwin

package main

func (*ModelEngine) restoreScheduledPrefix([]int32, *SequenceKVCache, Config) (int, int32) {
	return 0, 0
}

func (*ModelEngine) storeScheduledPrefix([]int32, *SequenceKVCache, int32, Config) {}
