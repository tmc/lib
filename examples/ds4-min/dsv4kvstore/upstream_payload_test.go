package dsv4kvstore

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
)

func upstreamPayloadFixture() []byte {
	b := make([]byte, upstreamPayloadHeaderSize+7)
	values := []uint32{
		upstreamPayloadMagic, upstreamPayloadVersion,
		8192, 2048, 128, 128, 2050, 2048, 43, 512, 128, 129280, 128,
	}
	for i, value := range values {
		binary.LittleEndian.PutUint32(b[i*4:], value)
	}
	copy(b[upstreamPayloadHeaderSize:], []byte("opaque"))
	return b
}

func TestInspectUpstreamPayload(t *testing.T) {
	data := upstreamPayloadFixture()
	got, err := InspectUpstreamPayload(bytes.NewReader(data), uint64(len(data)))
	if err != nil {
		t.Fatalf("InspectUpstreamPayload: %v", err)
	}
	if got.Context != 8192 || got.PrefillCapacity != 2048 ||
		got.RawCapacity != 128 || got.RawWindow != 128 ||
		got.CompressedCap != 2050 || got.Tokens != 2048 || got.Layers != 43 ||
		got.HeadDim != 512 || got.IndexerHeadDim != 128 ||
		got.Vocab != 129280 || got.RawLive != 128 {
		t.Fatalf("metadata = %+v", got)
	}
}

func TestInspectUpstreamPayloadRejectsHeaderMutations(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func([]byte)
	}{
		{"magic", func(b []byte) { binary.LittleEndian.PutUint32(b, 0) }},
		{"version", func(b []byte) { binary.LittleEndian.PutUint32(b[4:], 1) }},
		{"tokens-at-context", func(b []byte) { binary.LittleEndian.PutUint32(b[28:], 8192) }},
		{"raw-window-too-large", func(b []byte) { binary.LittleEndian.PutUint32(b[20:], 129) }},
		{"raw-live-does-not-match-window", func(b []byte) { binary.LittleEndian.PutUint32(b[48:], 127) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := upstreamPayloadFixture()
			test.mutate(data)
			_, err := InspectUpstreamPayload(bytes.NewReader(data), uint64(len(data)))
			if !errors.Is(err, ErrUpstreamPayloadHeader) {
				t.Fatalf("error = %v, want ErrUpstreamPayloadHeader", err)
			}
		})
	}
}

func TestInspectUpstreamPayloadDoesNotClaimRestore(t *testing.T) {
	data := upstreamPayloadFixture()
	if _, err := InspectUpstreamPayload(bytes.NewReader(data), 13*4-1); !errors.Is(err, ErrUpstreamTruncated) {
		t.Fatalf("short payload error = %v, want ErrUpstreamTruncated", err)
	}
	if !errors.Is(UpstreamPayloadError(), ErrUpstreamPayloadUnsupported) {
		t.Fatal("payload inspection unexpectedly changed restore capability")
	}
}

func smallLayoutFixture() []byte {
	const base = upstreamPayloadHeaderSize + 4 + 4 + 2*4 + 2*4
	// Layer 0: raw (4 bytes). Layer 1: raw, compressed, two attention
	// states, indexer, and two index states (268 bytes total).
	b := make([]byte, base+4+4+4+128+4+128)
	values := []uint32{
		upstreamPayloadMagic, upstreamPayloadVersion,
		4, 4, 1, 1, 4, 1, 2, 1, 1, 1, 1,
	}
	for i, value := range values {
		binary.LittleEndian.PutUint32(b[i*4:], value)
	}
	// n_comp[0:2] followed by n_index_comp[0:2].
	binary.LittleEndian.PutUint32(b[52+4+4+4:], 1)
	binary.LittleEndian.PutUint32(b[52+4+4+8+4:], 1)
	return b
}

func TestInspectUpstreamPayloadLayout(t *testing.T) {
	data := smallLayoutFixture()
	layout, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)), []uint32{0, 4})
	if err != nil {
		t.Fatalf("InspectUpstreamPayloadLayout: %v", err)
	}
	if layout.Bytes != uint64(len(data)) || len(layout.Sections) != 8 {
		t.Fatalf("layout = bytes %d sections %d, want %d and 8", layout.Bytes, len(layout.Sections), len(data))
	}
	if got := layout.Sections[len(layout.Sections)-1]; got.Offset+got.Bytes != uint64(len(data)) {
		t.Fatalf("last section ends at %d, want %d", got.Offset+got.Bytes, len(data))
	}
}

func TestInspectUpstreamPayloadLayoutRejectsRatioAndSize(t *testing.T) {
	data := smallLayoutFixture()
	if _, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)), []uint32{0}); err == nil {
		t.Fatal("short ratio list unexpectedly passed")
	}
	if _, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)), []uint32{0, 128}); err == nil {
		t.Fatal("mismatched ratio unexpectedly passed")
	}
	if _, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)-1), []uint32{0, 4}); err == nil {
		t.Fatal("truncated layout unexpectedly passed")
	}
}

func TestHashUpstreamPayloadTokenIDs(t *testing.T) {
	data := smallLayoutFixture()
	for i := uint32(0); i < 1; i++ {
		binary.LittleEndian.PutUint32(data[upstreamPayloadHeaderSize+int(i)*4:], i+17)
	}
	layout, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)), []uint32{0, 4})
	if err != nil {
		t.Fatal(err)
	}
	want := sha256.Sum256(data[upstreamPayloadHeaderSize : upstreamPayloadHeaderSize+layout.Metadata.Tokens*4])
	got, err := HashUpstreamPayloadTokenIDs(bytes.NewReader(data), layout)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("token ID digest = %x, want %x", got, want)
	}
	data[upstreamPayloadHeaderSize]++
	changed, err := HashUpstreamPayloadTokenIDs(bytes.NewReader(data), layout)
	if err != nil {
		t.Fatal(err)
	}
	if changed == got {
		t.Fatal("token ID mutation did not change digest")
	}
}

func TestHashUpstreamPayloadTokenIDsRejectsBounds(t *testing.T) {
	data := smallLayoutFixture()
	layout, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)), []uint32{0, 4})
	if err != nil {
		t.Fatal(err)
	}
	layout.Bytes = upstreamPayloadHeaderSize
	if _, err := HashUpstreamPayloadTokenIDs(bytes.NewReader(data), layout); err == nil {
		t.Fatal("truncated token section unexpectedly passed")
	}
	if _, err := HashUpstreamPayloadTokenIDs(nil, layout); err == nil {
		t.Fatal("nil reader unexpectedly passed")
	}
}

func TestValidateUpstreamRestoreContractRequiresProvenance(t *testing.T) {
	data := smallLayoutFixture()
	layout, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)), []uint32{0, 4})
	if err != nil {
		t.Fatal(err)
	}
	contract := UpstreamRestoreContract{
		ModelSHA256: [32]byte{1}, TokenIDsSHA256: [32]byte{2},
		PayloadABI: upstreamKVCPayloadABI, QuantBits: 2,
		Context: layout.Metadata.Context, PrefillCapacity: layout.Metadata.PrefillCapacity,
		CompressedCap: layout.Metadata.CompressedCap, Tokens: layout.Metadata.Tokens,
		Layers: layout.Metadata.Layers, HeadDim: layout.Metadata.HeadDim,
		IndexerHeadDim: layout.Metadata.IndexerHeadDim, Vocab: layout.Metadata.Vocab,
		UpstreamRawCapacity: layout.Metadata.RawCapacity,
		TargetRawCapacity:   layout.Metadata.RawLive, RawWindow: layout.Metadata.RawWindow,
		Ratios: []uint32{0, 4},
	}
	if err := ValidateUpstreamRestoreContract(layout, contract); err != nil {
		t.Fatalf("valid contract: %v", err)
	}
	contract.ModelSHA256 = [32]byte{}
	if err := ValidateUpstreamRestoreContract(layout, contract); err == nil {
		t.Fatal("missing model provenance unexpectedly passed")
	}
	contract.ModelSHA256 = [32]byte{1}
	contract.RawWindow++
	if err := ValidateUpstreamRestoreContract(layout, contract); err == nil {
		t.Fatal("geometry mutation unexpectedly passed")
	}
	layout.Metadata.RawCapacity = 4
	layout.Metadata.RawLive = 1
	contract.RawWindow = layout.Metadata.RawWindow
	contract.UpstreamRawCapacity = 4
	contract.TargetRawCapacity = 1
	if err := ValidateUpstreamRestoreContract(layout, contract); err != nil {
		t.Fatalf("distinct upstream and target raw capacities: %v", err)
	}
	contract.TargetRawCapacity = 4
	if err := ValidateUpstreamRestoreContract(layout, contract); err != nil {
		t.Fatalf("target capacity larger than live window: %v", err)
	}
	contract.TargetRawCapacity = 0
	if err := ValidateUpstreamRestoreContract(layout, contract); err == nil {
		t.Fatal("zero target capacity unexpectedly passed")
	}
}

func TestMapUpstreamRawRows(t *testing.T) {
	meta := UpstreamPayloadMetadata{Tokens: 10, RawLive: 4, HeadDim: 3}
	rows, err := MapUpstreamRawRows(meta, 8)
	if err != nil {
		t.Fatal(err)
	}
	want := []UpstreamRawRowMapping{
		{Position: 6, Slot: 6, Offset: 72},
		{Position: 7, Slot: 7, Offset: 84},
		{Position: 8, Slot: 0, Offset: 0},
		{Position: 9, Slot: 1, Offset: 12},
	}
	if !reflect.DeepEqual(rows, want) {
		t.Fatalf("rows = %+v, want %+v", rows, want)
	}
	meta.RawLive = 9
	if _, err := MapUpstreamRawRows(meta, 8); err == nil {
		t.Fatal("raw window larger than target ring unexpectedly passed")
	}
}

func TestReadUpstreamPayloadSectionBoundsAndLookup(t *testing.T) {
	data := smallLayoutFixture()
	layout, err := InspectUpstreamPayloadLayout(bytes.NewReader(data), uint64(len(data)), []uint32{0, 4})
	if err != nil {
		t.Fatal(err)
	}
	section := layout.Sections[0]
	dst := make([]byte, section.Bytes)
	if err := ReadUpstreamPayloadSection(bytes.NewReader(data), layout, section.Layer, section.Name, dst); err != nil {
		t.Fatal(err)
	}
	if err := ReadUpstreamPayloadSection(bytes.NewReader(data), layout, 99, "rawKV", nil); err == nil {
		t.Fatal("unknown section unexpectedly passed")
	}
	if err := ReadUpstreamPayloadSection(bytes.NewReader(data), layout, section.Layer, section.Name, dst[:len(dst)-1]); err == nil {
		t.Fatal("short destination unexpectedly passed")
	}
}

func TestValidateUpstreamRecordLayoutChecksEnvelopeAndFingerprint(t *testing.T) {
	payload := smallLayoutFixture()
	digest := sha256.Sum256(payload)
	meta, err := InspectUpstreamPayload(bytes.NewReader(payload), uint64(len(payload)))
	if err != nil {
		t.Fatal(err)
	}
	record := UpstreamRecord{
		Metadata: UpstreamMetadata{
			Context: meta.Context, Tokens: meta.Tokens,
			PayloadBytes: uint64(len(payload)),
		},
		Payload: payload, PayloadSHA256: digest,
	}
	if _, err := ValidateUpstreamRecordLayout(record, []uint32{0, 4}); err != nil {
		t.Fatalf("valid record: %v", err)
	}
	record.PayloadSHA256[0]++
	if _, err := ValidateUpstreamRecordLayout(record, []uint32{0, 4}); err == nil {
		t.Fatal("fingerprint mutation unexpectedly passed")
	}
}
