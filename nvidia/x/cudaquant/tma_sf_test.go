// SPDX-License-Identifier: Apache-2.0

package cudaquant

import "testing"

func TestAlignedStrides(t *testing.T) {
	rowCases := []struct {
		rows       int
		alignBytes int
		want       int
	}{
		{1, 0, 16},
		{16, 0, 16},
		{17, 16, 32},
		{5, 4, 8},
	}
	for _, tc := range rowCases {
		got, err := AlignedScaleRows(tc.rows, tc.alignBytes)
		if err != nil {
			t.Fatalf("AlignedScaleRows(%d, %d) failed: %v", tc.rows, tc.alignBytes, err)
		}
		if got != tc.want {
			t.Fatalf("AlignedScaleRows(%d, %d) = %d, want %d", tc.rows, tc.alignBytes, got, tc.want)
		}
	}
	if _, err := AlignedScaleRows(0, 16); err == nil {
		t.Fatal("AlignedScaleRows(0, 16) succeeded, want error")
	}
	if _, err := AlignedScaleRows(4, -1); err == nil {
		t.Fatal("AlignedScaleRows(4, -1) succeeded, want error")
	}

	strideCases := []struct {
		cols       int
		elemBytes  int
		alignBytes int
		want       int
	}{
		{50, 1, 0, 64},
		{64, 1, 32, 64},
		{20, 2, 32, 32},
		{10, 4, 32, 16},
	}
	for _, tc := range strideCases {
		got, err := AlignedOutputStride(tc.cols, tc.elemBytes, tc.alignBytes)
		if err != nil {
			t.Fatalf("AlignedOutputStride(%d, %d, %d) failed: %v", tc.cols, tc.elemBytes, tc.alignBytes, err)
		}
		if got != tc.want {
			t.Fatalf("AlignedOutputStride(%d, %d, %d) = %d, want %d", tc.cols, tc.elemBytes, tc.alignBytes, got, tc.want)
		}
	}
	if _, err := AlignedOutputStride(0, 1, 32); err == nil {
		t.Fatal("AlignedOutputStride(0, 1, 32) succeeded, want error")
	}
	if _, err := AlignedOutputStride(32, 0, 32); err == nil {
		t.Fatal("AlignedOutputStride(32, 0, 32) succeeded, want error")
	}
}

func TestTMAScaleLayoutTransform(t *testing.T) {
	rows, scaleCols := 5, 3
	alignedRows, err := AlignedScaleRows(rows, 16)
	if err != nil {
		t.Fatalf("AlignedScaleRows failed: %v", err)
	}
	if alignedRows != 16 {
		t.Fatalf("alignedRows = %d, want 16", alignedRows)
	}

	rowMajor := make([]uint32, rows*scaleCols)
	for r := 0; r < rows; r++ {
		for c := 0; c < scaleCols; c++ {
			rowMajor[r*scaleCols+c] = uint32((r+1)*100 + (c + 1))
		}
	}

	colMajor, err := TransformScaleLayoutColMajor(rowMajor, rows, scaleCols, alignedRows)
	if err != nil {
		t.Fatalf("TransformScaleLayoutColMajor failed: %v", err)
	}
	if len(colMajor) != scaleCols*alignedRows {
		t.Fatalf("len(colMajor) = %d, want %d", len(colMajor), scaleCols*alignedRows)
	}

	// Verify active entries and zero-filled padding rows against explicit 2D coordinates.
	for c := 0; c < scaleCols; c++ {
		for r := 0; r < alignedRows; r++ {
			got := colMajor[c*alignedRows+r]
			var want uint32
			if r < rows {
				want = uint32((r+1)*100 + (c + 1))
			}
			if got != want {
				t.Fatalf("colMajor[c=%d, r=%d] = %d, want %d", c, r, got, want)
			}
		}
	}

	recovered, err := InverseTransformScaleLayoutColMajor(colMajor, rows, scaleCols, alignedRows)
	if err != nil {
		t.Fatalf("InverseTransformScaleLayoutColMajor failed: %v", err)
	}
	for i, want := range rowMajor {
		if recovered[i] != want {
			t.Fatalf("recovered[%d] = %d, want %d", i, recovered[i], want)
		}
	}

	// Error cases.
	if _, err := TransformScaleLayoutColMajor(rowMajor, rows, scaleCols, rows-1); err == nil {
		t.Fatal("TransformScaleLayoutColMajor with alignedRows < rows succeeded")
	}
	if _, err := TransformScaleLayoutColMajor(rowMajor[:len(rowMajor)-1], rows, scaleCols, alignedRows); err == nil {
		t.Fatal("TransformScaleLayoutColMajor with short slice succeeded")
	}
	if _, err := InverseTransformScaleLayoutColMajor(colMajor[:len(colMajor)-1], rows, scaleCols, alignedRows); err == nil {
		t.Fatal("InverseTransformScaleLayoutColMajor with short slice succeeded")
	}
}
