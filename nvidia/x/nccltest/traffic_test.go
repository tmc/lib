package nccltest

import (
	"math"
	"slices"
	"testing"
)

func TestGeneratedTrafficSpread(t *testing.T) {
	for _, tt := range []struct {
		spread float64
		want   []int
	}{
		{0, []int{16, 16, 16, 16}},
		{0.5, []int{8, 12, 16, 24}},
		{1, []int{0, 8, 20, 32}},
	} {
		c := Case{Operation: AllToAllV, Bytes: 256, Spread: &tt.spread}
		l, err := c.Layout(4)
		if err != nil {
			t.Fatal(err)
		}
		for source := 0; source < 4; source++ {
			var got []int
			send, recv := 0, 0
			for distance := 0; distance < 4; distance++ {
				got = append(got, c.peerElements(l.ParamCount, 4, source, (source+distance)%4, 4))
				send += c.peerElements(l.ParamCount, 4, source, distance, 4)
				recv += c.peerElements(l.ParamCount, 4, distance, source, 4)
			}
			if !slices.Equal(got, tt.want) || send != l.SendCount || recv != l.RecvCount {
				t.Fatalf("spread=%g rank=%d counts=%v want=%v send=%d recv=%d layout=%+v", tt.spread, source, got, tt.want, send, recv, l)
			}
		}
	}
}

func TestSpreadValidation(t *testing.T) {
	for _, spread := range []float64{-1, 1.01, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if _, err := (Case{Operation: AllToAllV, Bytes: 256, Spread: &spread}).Layout(4); err == nil {
			t.Fatalf("accepted spread %g", spread)
		}
	}
	spread := 0.5
	if _, err := (Case{Operation: AllReduce, Bytes: 256, Spread: &spread}).Layout(4); err == nil {
		t.Fatal("spread accepted for another operation")
	}
}

func TestCaseCopySpread(t *testing.T) {
	spread := 0.5
	original := Case{Operation: AllToAllV, Spread: &spread}
	snapshot := original.copy()
	spread = 0
	if *snapshot.Spread != 0.5 {
		t.Fatal("caller changed snapshot")
	}
	result := snapshot.copy()
	*result.Spread = 1
	if *snapshot.Spread != 0.5 {
		t.Fatal("result changed next sample")
	}
}

func TestSpreadSourceSegments(t *testing.T) {
	for _, tt := range []struct {
		spread float64
		bytes  []int
	}{
		{0, []int{64, 64, 64, 64}},
		{0.5, []int{32, 48, 64, 96}},
		{1, []int{0, 32, 80, 128}},
	} {
		for _, datatype := range []DataType{Uint8, Float16, Float32, Float64} {
			c := Case{Operation: AllToAllV, DataType: datatype, Bytes: 256, Spread: &tt.spread}
			l, err := c.Layout(4)
			if err != nil {
				t.Fatal(err)
			}
			for destination := 0; destination < 4; destination++ {
				received := 0
				for source := 0; source < 4; source++ {
					sourceOffset := 0
					for peer := 0; peer < destination; peer++ {
						sourceOffset += tt.bytes[(peer-source+4)%4] / datatype.Size()
					}
					count := tt.bytes[(destination-source+4)%4] / datatype.Size()
					for i := 0; i < count; i++ {
						gotSource, gotElement, copy := sourceElement(c, l, 4, destination, received+i)
						if !copy || gotSource != source || gotElement != sourceOffset+i {
							t.Fatalf("spread=%g type=%s destination=%d index=%d: got (%d,%d,%v), want (%d,%d,true)", tt.spread, datatype, destination, received+i, gotSource, gotElement, copy, source, sourceOffset+i)
						}
					}
					received += count
				}
				if received != l.RecvCount {
					t.Fatalf("received=%d layout=%+v", received, l)
				}
			}
		}
	}
}

func TestSpreadCountOverflow(t *testing.T) {
	// On 64-bit hosts the weighted peer byte count rounds to 2^63 before
	// conversion to int; the conversion result must not determine validity.
	// On 32-bit hosts it remains exactly representable.
	if int(^uint(0)>>1) <= 1<<31-1 {
		t.Skip("requires 64-bit int")
	}
	c := Case{Operation: AllToAllV, DataType: Uint8, Bytes: int(^uint(0) >> 1)}
	if _, err := c.Layout(2); err == nil {
		t.Fatal("rounded overflowing peer count accepted")
	}
}
