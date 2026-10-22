package dsv4tools

import (
	"math"
	"testing"
)

func TestReferenceQ81AndMutations(t *testing.T) {
	values := make([]float32, 64)
	for i := range values {
		values[i] = float32(i-31) * 0.125
	}
	want, err := ReferenceQ81(values)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateQ81(values, want); err != nil {
		t.Fatal(err)
	}
	want[1].Quants[7]++
	if err := ValidateQ81(values, want); err == nil {
		t.Fatal("accepted mutated Q8_1 code")
	}
	if len(EncodeQ81(want)) != 72 {
		t.Fatal("unexpected Q8_1 encoding size")
	}
}

func TestReferenceQ81RejectsInvalidInputs(t *testing.T) {
	for _, values := range [][]float32{nil, make([]float32, 31), {1, 2, 3, 4}} {
		if _, err := ReferenceQ81(values); err == nil {
			t.Fatalf("accepted input length %d", len(values))
		}
	}
	values := make([]float32, 32)
	values[3] = float32(math.NaN())
	if _, err := ReferenceQ81(values); err == nil {
		t.Fatal("accepted non-finite Q8_1 input")
	}
}

func TestValidateExpertLayoutAndMutations(t *testing.T) {
	records := make([]ExpertLayoutRecord, 43)
	for i := range records {
		records[i] = ExpertLayoutRecord{Layer: i, GateType: "IQ2_XXS", UpType: "IQ2_XXS", DownType: "Q2_K", GateDims: []uint64{4096, 2048, 256}, UpDims: []uint64{4096, 2048, 256}, DownDims: []uint64{2048, 4096, 256}}
	}
	if err := ValidateExpertLayout(records); err != nil {
		t.Fatal(err)
	}
	records[7].DownType = "Q4_K"
	if err := ValidateExpertLayout(records); err == nil {
		t.Fatal("accepted type mutation")
	}
	records[7].DownType = "Q2_K"
	records[12].DownDims[0]++
	if err := ValidateExpertLayout(records); err == nil {
		t.Fatal("accepted dimension mutation")
	}
}

func TestQ4KReferenceAndMutation(t *testing.T) {
	block := make([]byte, Q4KBlockSize)
	block[0], block[1] = 0x00, 0x3c // d = 1
	for i := range block[4:16] {
		block[4+i] = 1
	}
	for i := range block[16:] {
		block[16+i] = 0x88
	}
	values, err := DequantizeQ4K(block)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range values {
		if value != 8 {
			t.Fatalf("Q4_K value %d = %g, want 8", i, value)
		}
	}
	input := make([]float32, Q4KBlockValues)
	for i := range input {
		input[i] = 1
	}
	got, err := ReferenceQ4K(block, input)
	if err != nil || got != 2048 {
		t.Fatalf("Q4_K dot = %g, %v, want 2048", got, err)
	}
	block[16] ^= 1
	mutated, err := DequantizeQ4K(block)
	if err != nil || mutated[0] == values[0] {
		t.Fatalf("Q4_K nibble mutation was not observed: %g %v", mutated[0], err)
	}
}

func TestQ4KReferenceCoversPackedHighScaleBits(t *testing.T) {
	block := make([]byte, Q4KBlockSize)
	block[0], block[1] = 0x00, 0x3c // d = 1
	block[2], block[3] = 0x00, 0x38 // dmin = 0.5
	// Group 4 reads the high two scale bits from scales[0] and
	// scales[4]. The low four bits come from scales[8].
	block[4] = 0xc0
	block[8] = 0x40
	block[12] = 0x21
	block[16+64] = 0x01
	values, err := DequantizeQ4K(block)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := values[4*32], float32(49)-float32(18)*0.5; got != want {
		t.Fatalf("group 4 first value = %g, want %g", got, want)
	}
}

func TestMXFP4ReferenceAndMutation(t *testing.T) {
	block := make([]byte, MXFP4BlockSize)
	block[0] = 127
	for i := range block[1:] {
		block[1+i] = 0xf6
	}
	values, err := DequantizeMXFP4(block)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 16; i++ {
		if values[i] != 4 || values[i+16] != -6 {
			t.Fatalf("MXFP4 values at %d = %g/%g, want 4/-6", i, values[i], values[i+16])
		}
	}
	input := make([]float32, MXFP4Values)
	for i := range input {
		input[i] = 1
	}
	got, err := ReferenceMXFP4(block, input)
	if err != nil || got != -32 {
		t.Fatalf("MXFP4 dot = %g, %v, want -32", got, err)
	}
	block[1] = 0x67
	mutated, err := DequantizeMXFP4(block)
	if err != nil || mutated[0] == values[0] {
		t.Fatalf("MXFP4 nibble mutation was not observed: %g %v", mutated[0], err)
	}
}

func TestMXFP4ReferenceCoversExponentAndNegativeZero(t *testing.T) {
	zeroExponent := make([]byte, MXFP4BlockSize)
	zeroExponent[0] = 0
	zeroExponent[1] = 0x01
	values, err := DequantizeMXFP4(zeroExponent)
	if err != nil {
		t.Fatal(err)
	}
	if values[0] == 0 || math.IsNaN(float64(values[0])) {
		t.Fatalf("E8M0 zero scale lost the subnormal value: %g", values[0])
	}

	maxExponent := make([]byte, MXFP4BlockSize)
	maxExponent[0] = 0xff
	maxExponent[1] = 0x21
	values, err = DequantizeMXFP4(maxExponent)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(float64(values[0]), 1) || !math.IsInf(float64(values[16]), 1) {
		t.Fatalf("E8M0 max scale = %g/%g, want positive infinity", values[0], values[16])
	}

	negativeZero := make([]byte, MXFP4BlockSize)
	negativeZero[0] = 127
	negativeZero[1] = 0x87
	values, err = DequantizeMXFP4(negativeZero)
	if err != nil {
		t.Fatal(err)
	}
	if math.Float32bits(values[16]) != 0x80000000 {
		t.Fatalf("negative zero nibble = %#x, want %#x", math.Float32bits(values[16]), uint32(0x80000000))
	}
}

func TestQuantReferencesRejectMalformedBlocks(t *testing.T) {
	if _, err := DequantizeQ4K(make([]byte, Q4KBlockSize-1)); err == nil {
		t.Fatal("accepted short Q4_K block")
	}
	if _, err := DequantizeMXFP4(make([]byte, MXFP4BlockSize+1)); err == nil {
		t.Fatal("accepted long MXFP4 block")
	}
	if _, err := ReferenceQ4K(make([]byte, Q4KBlockSize), make([]float32, 1)); err == nil {
		t.Fatal("accepted mismatched Q4_K input")
	}
	if _, err := ReferenceMXFP4(make([]byte, MXFP4BlockSize), make([]float32, 1)); err == nil {
		t.Fatal("accepted mismatched MXFP4 input")
	}
}
