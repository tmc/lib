package dsv41

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const payloadMagic = uint32(0x34565344)
const payloadVersion = uint32(2)

// WritePayload writes the pinned upstream V4.1 DSV4 payload layout. The file's
// complete SHA-256 must have been verified by BindVerified. The payload itself
// has no asset identity: callers must store AssetSHA256 in a separate sealed
// receipt. All state validation precedes the first write; writer errors can
// still leave a partial stream. It accepts completed nonempty frontiers only.
func (s *State) WritePayload(w io.Writer) error {
	if w == nil || !s.complete() || s.model.assetSHA == ([32]byte{}) || s.position == 0 || s.position >= s.context || len(s.tokens) != s.position || len(s.logits) != 129280 {
		return fmt.Errorf("V4.1 state has no verified complete payload frontier")
	}
	for _, token := range s.tokens {
		if token < 0 || token >= 129280 {
			return fmt.Errorf("invalid payload token")
		}
	}
	if !payloadFinite(s.logits) {
		return fmt.Errorf("non-finite payload logits")
	}
	for layer, raw := range s.kv.raw {
		n := min(s.position, 128) * 512
		if len(raw) < n || !payloadFinite(raw[:n]) {
			return fmt.Errorf("invalid payload raw layer %d", layer)
		}
	}
	for owner, o := range s.kv.owners {
		ratio := 2
		if owner == 3 {
			ratio = 1
		}
		if len(o.kv) != s.position/ratio*512 || len(o.index) != s.position/ratio*128 {
			return fmt.Errorf("invalid payload owner %d", owner)
		}
		if !payloadFinite(o.kv) || !payloadFinite(o.index) {
			return fmt.Errorf("non-finite payload owner %d", owner)
		}
		if owner < 3 && s.position&1 != 0 && (len(s.pool[owner].kv) != 512 || len(s.pool[owner].scores) != 512) {
			return fmt.Errorf("missing payload pool carry")
		}
		if owner < 3 && s.position&1 != 0 && (!payloadFinite(s.pool[owner].kv) || !payloadFinite(s.pool[owner].scores)) {
			return fmt.Errorf("non-finite payload carry")
		}
	}
	header := [13]uint32{payloadMagic, payloadVersion, uint32(s.context), 1, 128, 128, uint32(s.context + 1), uint32(s.position), 40, 512, 128, 129280, 0x413431}
	write := func(x any) error { return binary.Write(w, binary.LittleEndian, x) }
	if err := write(header); err != nil {
		return fmt.Errorf("write payload header: %w", err)
	}
	if err := write(s.tokens); err != nil {
		return fmt.Errorf("write payload tokens: %w", err)
	}
	if err := write(s.logits); err != nil {
		return fmt.Errorf("write payload logits: %w", err)
	}
	for _, raw := range s.kv.raw {
		if err := write(raw[:min(s.position, 128)*512]); err != nil {
			return fmt.Errorf("write payload raw KV: %w", err)
		}
	}
	for owner, o := range s.kv.owners {
		for _, values := range [][]float32{o.kv, o.index} {
			if err := write(values); err != nil {
				return fmt.Errorf("write payload owner: %w", err)
			}
		}
		if owner < 3 && s.position&1 != 0 {
			for _, values := range [][]float32{s.pool[owner].kv, s.pool[owner].scores} {
				if err := write(values); err != nil {
					return fmt.Errorf("write payload carry: %w", err)
				}
			}
		}
	}
	return nil
}

// RestorePayload reads exactly size bytes of the upstream payload and replaces
// the request only after the entire stream validates. asset must identify the
// original payload's complete model file, from an independently sealed receipt;
// it must match this Model's verified identity. The payload is not evidence of
// that identity by itself, nor does the asset receipt authenticate payload
// bytes: callers must verify a separately sealed payload checksum. The target
// context may differ, but must fit tokens. Non-finite state values are refused.
func (s *State) RestorePayload(r io.Reader, size int64, asset [32]byte) error {
	return s.restorePayload(r, size, asset, 0)
}

// sourceContext is zero for upstream raw payload imports, which may change
// context. A native envelope instead pins the original context explicitly.
func (s *State) restorePayload(r io.Reader, size int64, asset [32]byte, sourceContext int) error {
	if s == nil || s.model == nil || r == nil || size < 52 || asset == ([32]byte{}) || asset != s.model.assetSHA {
		return fmt.Errorf("unverified or incompatible V4.1 payload asset")
	}
	limited := &io.LimitedReader{R: r, N: size}
	read := func(x any) error { return binary.Read(limited, binary.LittleEndian, x) }
	readFloats := func(values []float32) error {
		if err := read(values); err != nil {
			return err
		}
		if !payloadFinite(values) {
			return fmt.Errorf("non-finite payload value")
		}
		return nil
	}
	var h [13]uint32
	if err := read(&h); err != nil {
		return fmt.Errorf("read payload header: %w", err)
	}
	pos := int(h[7])
	context := int(h[2])
	if h[0] != payloadMagic || h[1] != payloadVersion || context <= 0 || context > 1048576 || sourceContext != 0 && context != sourceContext || pos <= 0 || pos >= context || pos >= s.context || h[3] != 1 || h[4] != 128 || h[5] != 128 || h[6] != h[2]+1 || h[8] != 40 || h[9] != 512 || h[10] != 128 || h[11] != 129280 || h[12] != 0x413431 || size != payloadBytes(pos) {
		return fmt.Errorf("invalid V4.1 payload dimensions or size")
	}
	next, err := s.model.NewState(s.context)
	if err != nil {
		return err
	}
	next.tokens = make([]int32, pos)
	if err = read(next.tokens); err != nil {
		return fmt.Errorf("read payload tokens: %w", err)
	}
	for _, token := range next.tokens {
		if token < 0 || token >= 129280 {
			return fmt.Errorf("invalid payload token")
		}
	}
	next.logits = make([]float32, 129280)
	if err = readFloats(next.logits); err != nil {
		return fmt.Errorf("read payload logits: %w", err)
	}
	for _, raw := range next.kv.raw {
		if err = readFloats(raw[:min(pos, 128)*512]); err != nil {
			return fmt.Errorf("read payload raw KV: %w", err)
		}
	}
	for owner := range next.kv.owners {
		ratio := 2
		if owner == 3 {
			ratio = 1
		}
		o := &next.kv.owners[owner]
		o.kv = make([]float32, pos/ratio*512)
		o.index = make([]float32, pos/ratio*128)
		for _, values := range [][]float32{o.kv, o.index} {
			if err = readFloats(values); err != nil {
				return fmt.Errorf("read payload owner: %w", err)
			}
		}
		if owner < 3 {
			next.pool[owner].position = pos
			if pos&1 != 0 {
				next.pool[owner].kv = make([]float32, 512)
				next.pool[owner].scores = make([]float32, 512)
				for _, values := range [][]float32{next.pool[owner].kv, next.pool[owner].scores} {
					if err = readFloats(values); err != nil {
						return fmt.Errorf("read payload carry: %w", err)
					}
				}
			}
		}
	}
	if limited.N != 0 {
		return fmt.Errorf("payload extent not consumed")
	}
	// The source rebuilds the latest three mapped token IDs. Hashing those same
	// tokens reconstructs the private History value without opaque serialization.
	ids := make([]uint32, 48*min(pos, 3))
	if err = next.engram.layout.Hash(&next.engram.history, next.tokens[max(0, pos-3):], nil, ids); err != nil {
		return fmt.Errorf("restore payload history: %w", err)
	}
	next.position = pos
	next.kv.position = pos
	*s = *next
	return nil
}

func payloadFinite(values []float32) bool {
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

func payloadBytes(pos int) int64 {
	words := int64(13 + pos + 129280 + 40*min(pos, 128)*512)
	for owner := 0; owner < 4; owner++ {
		ratio := 2
		if owner == 3 {
			ratio = 1
		}
		words += int64(pos/ratio) * 640
		if owner < 3 && pos&1 != 0 {
			words += 1024
		}
	}
	return words * 4
}
