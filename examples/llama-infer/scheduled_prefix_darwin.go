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
