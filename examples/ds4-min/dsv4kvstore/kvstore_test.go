package dsv4kvstore

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func testRecord() Record {
	return Record{
		Header:  Header{QuantBits: QuantQ2, Context: 4096, RawWindow: 128, ModelDigest: [32]byte{1, 2, 3}},
		Tokens:  []int32{1, -2, 300},
		Payload: []byte{0, 1, 2, 255},
	}
}

func TestRoundTrip(t *testing.T) {
	want := testRecord()
	want.Header.Version = Version
	var buf bytes.Buffer
	if err := Encode(&buf, want); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	got, err := DecodeWithOptions(&buf, DecodeOptions{ExpectedQuantBits: QuantQ2, RejectDifferentQuant: true})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if got.Header != want.Header || !equalInt32(got.Tokens, want.Tokens) || !bytes.Equal(got.Payload, want.Payload) {
		t.Fatalf("round trip mismatch: got %#v, want %#v", got, want)
	}
}

func TestFormatSelfConsistency(t *testing.T) {
	want := testRecord()
	var first bytes.Buffer
	if err := Encode(&first, want); err != nil {
		t.Fatal(err)
	}
	got, err := DecodeWithOptions(bytes.NewReader(first.Bytes()), DecodeOptions{
		ExpectedQuantBits:       QuantQ2,
		RejectDifferentQuant:    true,
		ExpectedContext:         want.Header.Context,
		ExpectedRawWindow:       want.Header.RawWindow,
		RejectDifferentGeometry: true,
	})
	if err != nil {
		t.Fatalf("first decode: %v", err)
	}
	var second bytes.Buffer
	if err := Encode(&second, got); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("save-load-save changed the DSKV bytes")
	}
}

// TestSplitInvariant is the host-side G3 oracle. The envelope must not depend
// on how the underlying file writer or reader splits I/O. This catches code
// that accidentally treats a short write as a record boundary or assumes a
// single read fills a field.
func TestSplitInvariant(t *testing.T) {
	want := testRecord()
	want.Header.Version = Version
	var contiguous bytes.Buffer
	if err := Encode(&contiguous, want); err != nil {
		t.Fatal(err)
	}
	for _, size := range []int{1, 2, 3, 7, 16, 63} {
		t.Run("chunk-"+strconv.Itoa(size), func(t *testing.T) {
			var split bytes.Buffer
			if err := Encode(&chunkWriter{dst: &split, size: size}, want); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(split.Bytes(), contiguous.Bytes()) {
				t.Fatal("chunked writer changed the envelope")
			}
			got, err := Decode(&chunkReader{src: bytes.NewReader(split.Bytes()), size: size})
			if err != nil {
				t.Fatal(err)
			}
			if got.Header != want.Header || !equalInt32(got.Tokens, want.Tokens) || !bytes.Equal(got.Payload, want.Payload) {
				t.Fatalf("chunked reader changed the record: got %#v, want %#v", got, want)
			}
		})
	}
}

type chunkWriter struct {
	dst  io.Writer
	size int
}

func (w *chunkWriter) Write(p []byte) (int, error) {
	if len(p) > w.size {
		p = p[:w.size]
	}
	return w.dst.Write(p)
}

type chunkReader struct {
	src  io.Reader
	size int
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if len(p) > r.size {
		p = p[:r.size]
	}
	return r.src.Read(p)
}

func TestFormatSelfConsistencyRejectsTrailingData(t *testing.T) {
	var buf bytes.Buffer
	if err := Encode(&buf, testRecord()); err != nil {
		t.Fatal(err)
	}
	data := append(append([]byte(nil), buf.Bytes()...), 0)
	if _, err := Decode(bytes.NewReader(data)); !errors.Is(err, ErrTrailingData) {
		t.Fatalf("Decode appended data = %v, want ErrTrailingData", err)
	}
}

func TestFormatSelfConsistencyRejectsGeometryMutation(t *testing.T) {
	var buf bytes.Buffer
	if err := Encode(&buf, testRecord()); err != nil {
		t.Fatal(err)
	}
	data := append([]byte(nil), buf.Bytes()...)
	// raw_window is the little-endian u32 at offset 16.
	binary.LittleEndian.PutUint32(data[16:20], 64)
	_, err := DecodeWithOptions(bytes.NewReader(data), DecodeOptions{
		ExpectedContext:         4096,
		ExpectedRawWindow:       128,
		RejectDifferentGeometry: true,
	})
	if !errors.Is(err, ErrBadHeader) {
		t.Fatalf("Decode mutated raw_window = %v, want ErrBadHeader", err)
	}
}

func TestQuantMismatch(t *testing.T) {
	var buf bytes.Buffer
	if err := Encode(&buf, testRecord()); err != nil {
		t.Fatal(err)
	}
	_, err := DecodeWithOptions(&buf, DecodeOptions{ExpectedQuantBits: QuantQ4, RejectDifferentQuant: true})
	if !errors.Is(err, ErrQuantMismatch) {
		t.Fatalf("error = %v, want ErrQuantMismatch", err)
	}
}

func TestDecodeRejectsMutations(t *testing.T) {
	base := func() []byte {
		var buf bytes.Buffer
		if err := Encode(&buf, testRecord()); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	for _, test := range []struct {
		name   string
		mutate func([]byte)
		want   error
	}{
		{"magic", func(b []byte) { b[0] = 'X' }, ErrBadMagic},
		{"version", func(b []byte) { b[4] = 2 }, ErrBadVersion},
		{"truncated", func(b []byte) { b = b[:len(b)-1] }, ErrTruncated},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := base()
			if test.name == "truncated" {
				data = data[:len(data)-1]
			} else {
				test.mutate(data)
			}
			_, err := Decode(bytes.NewReader(data))
			if !errors.Is(err, test.want) {
				t.Fatalf("error = %v, want %v", err, test.want)
			}
		})
	}
}

func TestDecodeRejectsUpstreamKVC(t *testing.T) {
	data := make([]byte, headerSize)
	copy(data, []byte("KVC"))
	_, err := Decode(bytes.NewReader(data))
	if !errors.Is(err, ErrUpstreamFormat) || !errors.Is(err, ErrUpstreamPayloadUnsupported) {
		t.Fatalf("error = %v, want ErrUpstreamFormat", err)
	}
}

func upstreamKVCFixture() []byte {
	data := make([]byte, 48+4+3+5+2)
	copy(data, []byte("KVC"))
	data[3] = 1
	data[4] = QuantQ2
	data[5] = 2
	data[7] = 9
	binary.LittleEndian.PutUint32(data[8:12], 2048)
	binary.LittleEndian.PutUint32(data[12:16], 3)
	binary.LittleEndian.PutUint32(data[16:20], 4096)
	data[20] = 2
	binary.LittleEndian.PutUint64(data[24:32], 11)
	binary.LittleEndian.PutUint64(data[32:40], 12)
	binary.LittleEndian.PutUint64(data[40:48], 5)
	binary.LittleEndian.PutUint32(data[48:52], 3)
	copy(data[52:], []byte("abcPAYLOAD!!"))
	return data
}

func TestInspectUpstreamKVC(t *testing.T) {
	data := upstreamKVCFixture()
	got, err := InspectUpstream(bytes.NewReader(data), UpstreamInspectOptions{})
	if err != nil {
		t.Fatalf("InspectUpstream: %v", err)
	}
	if got.Version != 1 || got.QuantBits != QuantQ2 || got.ModelID != 9 ||
		got.Tokens != 2048 || got.Hits != 3 || got.Context != 4096 ||
		got.PayloadABI != 2 || got.TextBytes != 3 || got.PayloadBytes != 5 ||
		got.TrailerBytes != 2 || got.FileBytes != uint64(len(data)) {
		t.Fatalf("metadata = %+v", got)
	}
	if !errors.Is(UpstreamPayloadError(), ErrUpstreamPayloadUnsupported) {
		t.Fatal("upstream payload did not remain explicitly unsupported")
	}
}

func TestInspectUpstreamFileBoundsWithoutReadingPayload(t *testing.T) {
	data := upstreamKVCFixture()
	reader := &readerAtLimit{data: data, max: 52}
	got, err := InspectUpstreamFile(reader, uint64(len(data)), UpstreamInspectOptions{})
	if err != nil {
		t.Fatalf("InspectUpstreamFile: %v", err)
	}
	if got.FileBytes != uint64(len(data)) || got.TrailerBytes != 2 || reader.read != 52 {
		t.Fatalf("metadata = %+v, reads = %d, want file=%d trailer=2 reads=52", got, reader.read, len(data))
	}
}

func TestReadUpstreamCopiesOpaqueEnvelope(t *testing.T) {
	data := upstreamFixture(t, []byte("prompt"), []byte{1, 2, 3}, []byte{9, 8})
	record, err := ReadUpstream(bytes.NewReader(data), UpstreamInspectOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if string(record.Text) != "prompt" || !bytes.Equal(record.Payload, []byte{1, 2, 3}) || !bytes.Equal(record.Trailer, []byte{9, 8}) {
		t.Fatalf("record = %#v", record)
	}
	if record.Metadata.FileBytes != uint64(len(data)) || record.Metadata.TrailerBytes != 2 {
		t.Fatalf("metadata = %#v", record.Metadata)
	}
	wantDigest := sha256.Sum256([]byte{1, 2, 3})
	if record.PayloadSHA256 != wantDigest {
		t.Fatalf("payload digest = %x, want %x", record.PayloadSHA256, wantDigest)
	}
}

func TestReadUpstreamRejectsPayloadBeforeAllocation(t *testing.T) {
	data := upstreamFixture(t, []byte("prompt"), []byte{1, 2, 3}, nil)
	_, err := ReadUpstream(bytes.NewReader(data), UpstreamInspectOptions{MaxPayloadBytes: 2})
	if !errors.Is(err, ErrUpstreamTooLarge) {
		t.Fatalf("ReadUpstream error = %v, want payload bound", err)
	}
}

func TestReadUpstreamPathRejectsTrailingTruncation(t *testing.T) {
	data := upstreamFixture(t, []byte("prompt"), []byte{1, 2, 3}, nil)
	dir := t.TempDir()
	path := filepath.Join(dir, "cache.kvc")
	if err := os.WriteFile(path, data[:len(data)-1], 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadUpstreamPath(path, UpstreamInspectOptions{}); !errors.Is(err, ErrUpstreamTruncated) {
		t.Fatalf("ReadUpstreamPath error = %v, want truncation", err)
	}
}

func upstreamFixture(t *testing.T, text, payload, trailer []byte) []byte {
	t.Helper()
	h := make([]byte, upstreamKVCHeaderSize)
	copy(h[:3], []byte("KVC"))
	h[3], h[4], h[20] = 1, QuantQ2, upstreamKVCPayloadABI
	binary.LittleEndian.PutUint32(h[8:12], 2)
	binary.LittleEndian.PutUint32(h[16:20], 4096)
	binary.LittleEndian.PutUint64(h[40:48], uint64(len(payload)))
	out := append([]byte(nil), h...)
	var n [4]byte
	binary.LittleEndian.PutUint32(n[:], uint32(len(text)))
	out = append(out, n[:]...)
	out = append(out, text...)
	out = append(out, payload...)
	out = append(out, trailer...)
	return out
}

func TestInspectUpstreamFileRejectsDeclaredBodyPastFile(t *testing.T) {
	data := upstreamKVCFixture()
	short := data[:upstreamKVCHeaderSize+4+3+5-1]
	_, err := InspectUpstreamFile(bytes.NewReader(short), uint64(len(short)), UpstreamInspectOptions{})
	if !errors.Is(err, ErrUpstreamTruncated) {
		t.Fatalf("InspectUpstreamFile = %v, want %v", err, ErrUpstreamTruncated)
	}
}

func TestInspectUpstreamFileRejectsOversizedFileBeforeRead(t *testing.T) {
	reader := &readerAtLimit{data: upstreamKVCFixture(), max: 0}
	_, err := InspectUpstreamFile(reader, uint64(len(upstreamKVCFixture())), UpstreamInspectOptions{MaxFileBytes: 1})
	if !errors.Is(err, ErrUpstreamTooLarge) {
		t.Fatalf("InspectUpstreamFile = %v, want %v", err, ErrUpstreamTooLarge)
	}
	if reader.read != 0 {
		t.Fatalf("read %d bytes before rejecting oversized file", reader.read)
	}
}

func TestInspectUpstreamPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.kvc")
	data := upstreamKVCFixture()
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	got, err := InspectUpstreamPath(path, UpstreamInspectOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got.FileBytes != uint64(len(data)) || got.PayloadBytes != 5 || got.TrailerBytes != 2 {
		t.Fatalf("metadata = %+v, want file=%d payload=5 trailer=2", got, len(data))
	}
}

func TestInspectUpstreamPathRejectsBeforeReadingOversizedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.kvc")
	if err := os.WriteFile(path, upstreamKVCFixture(), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := InspectUpstreamPath(path, UpstreamInspectOptions{MaxFileBytes: 1}); !errors.Is(err, ErrUpstreamTooLarge) {
		t.Fatalf("InspectUpstreamPath oversized error = %v, want %v", err, ErrUpstreamTooLarge)
	}
}

func TestInspectUpstreamPathRejectsMissingAndNonRegular(t *testing.T) {
	if _, err := InspectUpstreamPath("", UpstreamInspectOptions{}); err == nil {
		t.Fatal("empty path unexpectedly accepted")
	}
	dir := t.TempDir()
	if _, err := InspectUpstreamPath(dir, UpstreamInspectOptions{}); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("directory error = %v, want non-regular-file error", err)
	}
}

type readerAtLimit struct {
	data []byte
	max  int
	read int
}

func (r *readerAtLimit) ReadAt(p []byte, off int64) (int, error) {
	if r.max == 0 {
		return 0, io.EOF
	}
	if off < 0 || off >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[off:])
	if n > r.max {
		n = r.max
	}
	r.read += n
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func TestInspectUpstreamKVCRejectsMutations(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func([]byte)
		want   error
	}{
		{"version", func(b []byte) { b[3] = 2 }, ErrUpstreamBadVersion},
		{"payload ABI", func(b []byte) { b[20] = 1 }, ErrUpstreamBadPayloadABI},
		{"quant", func(b []byte) { b[4] = 8 }, ErrUpstreamBadQuant},
		{"tokens", func(b []byte) { binary.LittleEndian.PutUint32(b[8:12], 0) }, ErrUpstreamBadHeader},
		{"payload bound", func(b []byte) { binary.LittleEndian.PutUint64(b[40:48], 6) }, ErrUpstreamTooLarge},
		{"truncated text", func(b []byte) { b[48] = 10 }, ErrUpstreamTruncated},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := append([]byte(nil), upstreamKVCFixture()...)
			test.mutate(data)
			options := UpstreamInspectOptions{}
			if test.name == "payload bound" {
				options.MaxPayloadBytes = 5
			}
			_, err := InspectUpstream(bytes.NewReader(data), options)
			if !errors.Is(err, test.want) {
				t.Fatalf("InspectUpstream = %v, want %v", err, test.want)
			}
		})
	}
}

func TestInspectUpstreamKVCRejectsFileBound(t *testing.T) {
	_, err := InspectUpstream(bytes.NewReader(upstreamKVCFixture()), UpstreamInspectOptions{MaxFileBytes: 60})
	if !errors.Is(err, ErrUpstreamTooLarge) {
		t.Fatalf("InspectUpstream = %v, want ErrUpstreamTooLarge", err)
	}
}

func TestDecodeBoundsPayload(t *testing.T) {
	var buf bytes.Buffer
	if err := Encode(&buf, testRecord()); err != nil {
		t.Fatal(err)
	}
	_, err := DecodeWithOptions(bytes.NewReader(buf.Bytes()), DecodeOptions{MaxPayload: 1})
	if !errors.Is(err, ErrTooLarge) {
		t.Fatalf("error = %v, want ErrTooLarge", err)
	}
}

func TestKVCBudgetAccountsForAllBytesAndSlack(t *testing.T) {
	ok, fileBytes, required, err := KVCFitsBudget(83, 20, 10, 1)
	if err != nil {
		t.Fatal(err)
	}
	if ok || fileBytes != UpstreamKVCFixedBytes+31 || required != fileBytes+1 {
		t.Fatalf("KVCFitsBudget = %v, file=%d, required=%d", ok, fileBytes, required)
	}
	ok, fileBytes, required, err = KVCFitsBudget(required, 20, 10, 1)
	if err != nil || !ok || fileBytes != UpstreamKVCFixedBytes+31 {
		t.Fatalf("budget at reservation = %v, file=%d, required=%d, err=%v", ok, fileBytes, required, err)
	}
	if ok, _, _, err := KVCFitsBudget(1, 0, 0, 0); err != nil || ok {
		t.Fatalf("zero-byte entry unexpectedly fit: ok=%v err=%v", ok, err)
	}
	if ok, fileBytes, required, err := KVCFitsBudget(0, 0, 0, 0); err != nil || !ok || fileBytes != UpstreamKVCFixedBytes || required != UpstreamKVCFixedBytes+1 {
		t.Fatalf("unlimited budget = %v, file=%d, required=%d, err=%v", ok, fileBytes, required, err)
	}
}

func TestKVCBudgetRejectsOverflowMutations(t *testing.T) {
	max := ^uint64(0)
	for _, test := range []struct {
		name                   string
		text, payload, trailer uint64
	}{
		{"text", max, 0, 0},
		{"payload", 0, max, 0},
		{"trailer", 0, 0, max},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := KVCSize(test.text, test.payload, test.trailer); !errors.Is(err, ErrBudgetOverflow) {
				t.Fatalf("KVCSize overflow = %v, want %v", err, ErrBudgetOverflow)
			}
		})
	}
	if _, err := KVCBudgetRequired(max); !errors.Is(err, ErrBudgetOverflow) {
		t.Fatalf("KVCBudgetRequired overflow = %v, want %v", err, ErrBudgetOverflow)
	}
}

func TestKVCBudgetCountsTrailerMutation(t *testing.T) {
	without, err := KVCSize(8, 16, 0)
	if err != nil {
		t.Fatal(err)
	}
	with, err := KVCSize(8, 16, 1)
	if err != nil {
		t.Fatal(err)
	}
	if with != without+1 {
		t.Fatalf("trailer was not counted: without=%d with=%d", without, with)
	}
}

func TestStoreLenUsesAlignedBoundaries(t *testing.T) {
	for _, test := range []struct {
		name   string
		tokens int
		want   int
	}{
		{"empty", 0, 0},
		{"below minimum", 511, 511},
		{"minimum", 512, 512},
		{"trim margin", 2047, 2047},
		{"first boundary", 2080, 2048},
		{"between boundaries", 4095, 2048},
		{"second boundary", 4128, 4096},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := StoreLen(test.tokens, Options{}); got != test.want {
				t.Fatalf("StoreLen(%d) = %d, want %d", test.tokens, got, test.want)
			}
		})
	}
}

func TestStoreLenHonorsPolicyAndRejectsBoundaryMutations(t *testing.T) {
	options := Options{MinTokens: 4, BoundaryTrimTokens: 2, BoundaryAlignTokens: 8}
	for _, test := range []struct {
		name   string
		tokens int
		want   int
	}{
		{"custom minimum", 5, 5},
		{"custom first boundary", 10, 8},
		{"custom second boundary", 18, 16},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := StoreLen(test.tokens, options); got != test.want {
				t.Fatalf("StoreLen(%d) = %d, want %d", test.tokens, got, test.want)
			}
		})
	}
	if got := StoreLen(17, Options{MinTokens: 4, BoundaryTrimTokens: 1, BoundaryAlignTokens: 8}); got == 8 {
		t.Fatal("StoreLen ignored the trim policy mutation")
	}
	if got := StoreLen(15, Options{MinTokens: 4, BoundaryTrimTokens: 2, BoundaryAlignTokens: 4}); got == 8 {
		t.Fatal("StoreLen ignored the alignment policy mutation")
	}
}

func TestChatAnchorStopsAtAssistantAndUsesLastUserMarker(t *testing.T) {
	tokens := make([]int32, 12)
	for i := range tokens {
		tokens[i] = int32(i)
	}
	tokens[3], tokens[7], tokens[10] = 99, 99, 100
	if got := ChatAnchor(tokens, 99, 100, Options{MinTokens: 4}); got != 7 {
		t.Fatalf("ChatAnchor = %d, want 7", got)
	}
	if got := ChatAnchor(tokens, 99, 101, Options{MinTokens: 4}); got != 7 {
		t.Fatalf("ChatAnchor without assistant marker = %d, want 7", got)
	}
}

func TestChatAnchorRejectsShortAndInvalidMarkers(t *testing.T) {
	tokens := []int32{9, 9, 4, 9, 8}
	for _, test := range []struct {
		name       string
		user, asst int
		options    Options
	}{
		{name: "short", user: 9, asst: 8, options: Options{MinTokens: 4}},
		{name: "negative user", user: -1, asst: 8},
		{name: "negative assistant", user: 9, asst: -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := ChatAnchor(tokens, test.user, test.asst, test.options); got != -1 {
				t.Fatalf("ChatAnchor = %d, want -1", got)
			}
		})
	}
	if got := ChatAnchor([]int32{9, 9, 9, 8, 9}, 9, 8, Options{MinTokens: 1}); got != 2 {
		t.Fatalf("ChatAnchor before assistant = %d, want 2", got)
	}
}

func TestDefaultOptionsMatchUpstreamPolicy(t *testing.T) {
	got := DefaultOptions()
	want := Options{
		BudgetMB:            4096,
		MinTokens:           512,
		ColdMaxTokens:       30000,
		ContinuedInterval:   10000,
		BoundaryTrimTokens:  32,
		BoundaryAlignTokens: 2048,
	}
	if got != want {
		t.Fatalf("DefaultOptions() = %+v, want %+v", got, want)
	}
	got.BudgetMB++
	if DefaultOptions().BudgetMB != want.BudgetMB {
		t.Fatal("DefaultOptions returned shared mutable state")
	}
}

func TestContinuedStoreTarget(t *testing.T) {
	options := DefaultOptions()
	for _, test := range []struct {
		name       string
		live, last int
		want       int
	}{
		{"below minimum", 512, 0, 0},
		{"first aligned target", 2048, 0, 0},
		{"continued target", 10240, 0, 10240},
		{"already stored", 10240, 10240, 0},
		{"not aligned", 12288, 10240, 0},
		{"next target", 20480, 10240, 20480},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := ContinuedStoreTarget(test.live, test.last, options); got != test.want {
				t.Fatalf("ContinuedStoreTarget(%d, %d) = %d, want %d", test.live, test.last, got, test.want)
			}
		})
	}
	if got := ContinuedStoreTarget(10000, 0, Options{ContinuedInterval: 10000, BoundaryAlignTokens: 2048, MinTokens: 512}); got != 0 {
		t.Fatalf("unaligned interval produced target %d", got)
	}
	if got := ContinuedStoreTarget(10240, 0, Options{}); got != 0 {
		t.Fatalf("disabled interval produced target %d", got)
	}
}

func TestEvictionScorePolicy(t *testing.T) {
	when := time.Date(2026, 9, 30, 12, 0, 0, 0, time.UTC)
	base := EvictionMetadata{
		Reason: ReasonUnknown, Hits: 3, Tokens: 2048, FileBytes: 1024,
		LastUsed: when, Now: when,
	}
	if got := EvictionScore(base); got != 8 {
		t.Fatalf("EvictionScore at age zero = %v, want 8", got)
	}
	aged := base
	aged.LastUsed = when.Add(-6 * time.Hour)
	if got := EvictionScore(aged); got != 5 {
		t.Fatalf("EvictionScore after one half-life = %v, want 5", got)
	}
	for _, reason := range []CacheReason{ReasonCold, ReasonEvict, ReasonShutdown} {
		meta := base
		meta.Reason = reason
		if got := EvictionScore(meta); got != 16 {
			t.Fatalf("EvictionScore(%d) = %v, want 16", reason, got)
		}
	}
	continued := base
	continued.Reason = ReasonContinued
	continued.ContinuedSuperseded = true
	if got := EvictionScore(continued); got != 0.3875 {
		t.Fatalf("superseded continued score = %v, want 0.3875", got)
	}
}

func TestEvictionScoreRejectsMissingSizeAndClampsClock(t *testing.T) {
	when := time.Date(2026, 9, 30, 12, 0, 0, 0, time.UTC)
	base := EvictionMetadata{Hits: 1, Tokens: 1, FileBytes: 1, LastUsed: when, Now: when}
	for _, test := range []struct {
		name string
		edit func(*EvictionMetadata)
	}{
		{"missing tokens", func(m *EvictionMetadata) { m.Tokens = 0 }},
		{"missing file", func(m *EvictionMetadata) { m.FileBytes = 0 }},
		{"missing timestamp", func(m *EvictionMetadata) { m.LastUsed = time.Time{} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			meta := base
			test.edit(&meta)
			if got := EvictionScore(meta); got != 0 {
				t.Fatalf("EvictionScore = %v, want zero", got)
			}
		})
	}
	future := base
	future.LastUsed = when.Add(time.Hour)
	if got, want := EvictionScore(future), EvictionScore(base); got != want {
		t.Fatalf("future last-used score = %v, want %v", got, want)
	}
}

func equalInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
