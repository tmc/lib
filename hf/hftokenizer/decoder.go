package hftokenizer

import (
	"fmt"
	"unicode/utf8"
)

// Decoder converts a sequence of token-ID chunks into text. It retains incomplete
// UTF-8 between calls. Use Tokenizer.NewDecoder; the zero value is not usable.
// A Decoder must not be used concurrently.
type Decoder struct {
	tokenizer *Tokenizer
	pending   [utf8.UTFMax - 1]byte
	n         int
}

// NewDecoder returns an incremental text decoder using the tokenizer's vocabulary.
// Create a separate decoder for each independent or interrupted token stream.
func (t *Tokenizer) NewDecoder() *Decoder { return &Decoder{tokenizer: t} }

// Decode returns the complete text available after the supplied IDs. Special
// tokens are omitted. With final true, an incomplete final UTF-8 sequence becomes
// U+FFFD and the pending buffer is cleared. An error leaves the buffer unchanged.
// Concatenating the results through final=true matches Tokenizer.Decode on the
// complete sequence. A flushed decoder can be used for a new sequence.
func (d *Decoder) Decode(ids []int32, final bool) (string, error) {
	if d == nil || d.tokenizer == nil {
		return "", fmt.Errorf("tokenizer: uninitialized decoder")
	}
	raw, err := d.tokenizer.DecodeBytes(ids)
	if err != nil {
		return "", err
	}
	data := make([]byte, 0, d.n+len(raw))
	data = append(data, d.pending[:d.n]...)
	data = append(data, raw...)
	end := len(data)
	if !final {
		end = 0
		for end < len(data) && utf8.FullRune(data[end:]) {
			_, n := utf8.DecodeRune(data[end:])
			end += n
		}
	}
	text := lossyUTF8(data[:end])
	d.n = copy(d.pending[:], data[end:])
	return text, nil
}
