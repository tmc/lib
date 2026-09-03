//go:build darwin

package main

func (eng *ModelEngine) restoreScheduledPrefix(tokens []int32, sequence *SequenceKVCache, cfg Config) (int, int32) {
	if snapshot, matched, ok := eng.prefixes.load(tokens); ok && sequence.RestorePrefixPartial(snapshot, matched, cfg) {
		if matched == len(tokens) && snapshot.tokens == matched {
			return matched, snapshot.current
		}
		return matched, 0
	}
	return 0, 0
}

func (eng *ModelEngine) storeScheduledPrefix(tokens []int32, sequence *SequenceKVCache, current int32, cfg Config) {
	eng.prefixes.store(tokens, sequence.CapturePrefix(len(tokens), current, cfg))
}

func (eng *ModelEngine) restoreScheduledPrefixRow(tokens []int32, cache *BatchedAttentionCache, row int) (int, int32) {
	if snapshot, matched, ok := eng.prefixes.load(tokens); ok && cache.restorePrefix(row, matched, snapshot) {
		if matched == len(tokens) && snapshot.tokens == matched {
			return matched, snapshot.current
		}
		return matched, 0
	}
	return 0, 0
}

func (eng *ModelEngine) storeScheduledPrefixRow(tokens []int32, cache *BatchedAttentionCache, row int, current int32) {
	eng.prefixes.store(tokens, cache.capturePrefix(row, len(tokens), current))
}
