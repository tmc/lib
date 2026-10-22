//go:build linux

package main

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// TestCUDAQ81MatvecAlignedBits gates ds4_q8_0_matvec_q8_1_aligned against
// ds4_q8_0_matvec_q8_1 on the same weights, and requires the two to agree on
// every bit rather than within a tolerance.
//
// A tolerance would not be a gate here. The aligned kernel exists to move the
// weights from the model mapping into the dense artifact's device memory, and
// the argument that it is bit-exact is an argument about text -- same b = lane,
// same b += 32 stride, same eight __dp4a steps, same accumulation line, same
// shuffle tree. Register pressure is the one thing that argument cannot pin,
// and a 1 ULP difference is exactly what it would produce. The decode digest
// cannot see that: argmax absorbs it. So the comparison is on raw float bits.
//
// The positive controls matter as much as the comparison. Two kernels reading
// two buffers can agree because both are zero, or because the repack silently
// wrote nothing; perturbing one byte of each plane proves the comparison is
// live and that both planes are actually read.
func TestCUDAQ81MatvecAlignedBits(t *testing.T) {
	// cols must be a multiple of 32 for Q8_0 and rows a multiple of 8 so the
	// last block is full; the shapes below bracket the decode population.
	for _, shape := range []struct{ rows, cols int }{
		{128, 1024}, {128, 4096}, {512, 4096}, {2048, 4096},
		// The output projection: 129280 rows is the vocabulary, and the one
		// shape whose grid (16160 blocks) dwarfs the 48 SMs.
		{129280, 4096},
	} {
		t.Run(strconv.Itoa(shape.rows)+"x"+strconv.Itoa(shape.cols), func(t *testing.T) {
			r := cudaSynthFoundation(t)
			rows, cols := shape.rows, shape.cols
			weights := cudaSynthQ80(rand.New(rand.NewSource(int64(rows*cols))), rows, cols, -8)
			tensor := cudaSynthTensor(t, r, "aligned Q8_0 fixture", 8, []uint64{uint64(cols), uint64(rows)}, weights)

			x := make([]float32, cols)
			for i := range x {
				x[i] = float32((i*37+i/17)%257-128) / 128
			}
			raw := cudaQ81ActivationReference(x)
			in := cudaSynthUpload(t, r, x)
			q := cudaExpertAllocBytes(t, r, len(raw))
			if err := r.encodeQuantQ81(in, q, cols); err != nil {
				t.Fatal(err)
			}

			run := func(f func(out cudamem.Slice[float32]) error) []float32 {
				t.Helper()
				out := cudaTestAlloc(t, r, rows)
				poison := make([]float32, rows)
				for i := range poison {
					poison[i] = float32(math.NaN())
				}
				if err := out.CopyFromHost(poison); err != nil {
					t.Fatal(err)
				}
				if err := f(out); err != nil {
					t.Fatal(err)
				}
				if err := syncStream(r.stream()); err != nil {
					t.Fatal(err)
				}
				return cudaTestRead(t, out)
			}

			want := run(func(out cudamem.Slice[float32]) error {
				return r.encodeQ80MatvecQ81(tensor, q, out)
			})

			// The artifact, built the way buildDenseArtifact builds it.
			alignedBytes := denseQ8AlignedBytes(rows, cols)
			if alignedBytes == 0 {
				t.Fatalf("denseQ8AlignedBytes(%d, %d) = 0", rows, cols)
			}
			aligned := cudaExpertAllocBytes(t, r, alignedBytes)
			nblk := uint64(rows) * uint64(cols/32)
			if err := r.repackQ80(aligned, tensor.device, nblk); err != nil {
				t.Fatal(err)
			}
			if err := syncStream(r.stream()); err != nil {
				t.Fatal(err)
			}

			got := run(func(out cudamem.Slice[float32]) error {
				return r.encodeQ80MatvecQ81Aligned(aligned, uint64(cols), uint64(rows), q, out)
			})

			// The comparison must be able to fail on data, not just on shape:
			// an all-zero or all-NaN pair would agree bit for bit.
			var nonzero int
			for _, v := range want {
				if v != 0 && !math.IsNaN(float64(v)) {
					nonzero++
				}
			}
			if nonzero < rows/2 {
				t.Fatalf("reference output is degenerate: only %d of %d rows are finite and nonzero", nonzero, rows)
			}

			diff := cudaQ81AlignedBitDiff(got, want)
			if diff >= 0 {
				t.Fatalf("aligned kernel differs from the raw-block kernel at row %d: %v (0x%08x) vs %v (0x%08x)",
					diff, got[diff], math.Float32bits(got[diff]), want[diff], math.Float32bits(want[diff]))
			}

			// Positive controls. Perturbing one byte of each plane must break
			// the agreement; if it does not, that plane is not being read and
			// the green above is vacuous.
			dqBytes := int((nblk*2 + 63) &^ 63)
			for _, probe := range []struct {
				name   string
				offset int
			}{
				{"dq plane", 0},
				{"qs plane", dqBytes},
			} {
				t.Run(probe.name, func(t *testing.T) {
					orig := make([]byte, 1)
					one := aligned.Slice(probe.offset, probe.offset+1)
					if err := one.CopyToHost(orig); err != nil {
						t.Fatal(err)
					}
					if err := one.CopyFromHost([]byte{orig[0] ^ 0x40}); err != nil {
						t.Fatal(err)
					}
					defer func() {
						if err := one.CopyFromHost(orig); err != nil {
							t.Fatal(err)
						}
					}()
					perturbed := run(func(out cudamem.Slice[float32]) error {
						return r.encodeQ80MatvecQ81Aligned(aligned, uint64(cols), uint64(rows), q, out)
					})
					if cudaQ81AlignedBitDiff(perturbed, want) < 0 {
						t.Fatalf("flipping a byte of the %s changed nothing: the comparison does not read it", probe.name)
					}
				})
			}
		})
	}
}

// cudaQ81AlignedBitDiff returns the first index at which a and b differ in
// bits, or -1 when they are identical. NaN compares by bits here on purpose:
// the poison value is NaN and an untouched output must show up as a
// difference, not as an incomparable.
func cudaQ81AlignedBitDiff(a, b []float32) int {
	if len(a) != len(b) {
		return 0
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return i
		}
	}
	return -1
}

// TestCUDAQ80MatvecQuantizedAlignedBits is TestCUDAQ81MatvecAlignedBits for the
// float-activation pair: ds4_q8_0_matvec_quantized_aligned against
// ds4_q8_0_matvec_quantized, and the block-diagonal variant against its own
// raw-block original. Same reasoning, same positive controls.
//
// The block-diagonal case earns its own arm because the artifact is not one
// plane there. buildDenseArtifact repacks each of the eight diagonal blocks
// separately and lays the planes end to end, so the aligned kernel addresses a
// diagonal by a byte stride where the raw-block kernel addresses it by a row
// offset. That is an index change the bit comparison can catch and the
// text-level argument cannot.
func TestCUDAQ80MatvecQuantizedAlignedBits(t *testing.T) {
	for _, shape := range []struct {
		rows, cols, diag int
	}{
		{2048, 4096, 1}, // sharedDown and attnOutB
		{4096, 8192, 1}, // attnOutB at the model width
		{8192, 4096, 8}, // attnOutA, eight diagonal blocks of 1024
	} {
		name := strconv.Itoa(shape.rows) + "x" + strconv.Itoa(shape.cols)
		if shape.diag > 1 {
			name += "/diag" + strconv.Itoa(shape.diag)
		}
		t.Run(name, func(t *testing.T) {
			r := cudaSynthFoundation(t)
			rows, cols, diag := shape.rows, shape.cols, shape.diag
			perDiag := rows / diag
			weights := cudaSynthQ80(rand.New(rand.NewSource(int64(rows*cols+diag))), rows, cols, -8)
			tensor := cudaSynthTensor(t, r, "aligned Q8_0 quantized fixture", 8, []uint64{uint64(cols), uint64(rows)}, weights)

			x := make([]float32, cols*diag)
			for i := range x {
				x[i] = float32((i*37+i/17)%257-128) / 128
			}
			in := cudaSynthUpload(t, r, x)
			q := cudaExpertAllocBytes(t, r, len(x)/32*36)
			if err := r.encodeQuantQ80(in, q, cols); err != nil {
				t.Fatal(err)
			}

			run := func(f func(out cudamem.Slice[float32]) error) []float32 {
				t.Helper()
				out := cudaTestAlloc(t, r, rows)
				poison := make([]float32, rows)
				for i := range poison {
					poison[i] = float32(math.NaN())
				}
				if err := out.CopyFromHost(poison); err != nil {
					t.Fatal(err)
				}
				if err := f(out); err != nil {
					t.Fatal(err)
				}
				if err := syncStream(r.stream()); err != nil {
					t.Fatal(err)
				}
				return cudaTestRead(t, out)
			}

			want := run(func(out cudamem.Slice[float32]) error {
				if diag > 1 {
					return r.encodeQ80MatvecQuantizedBlockDiag(tensor, q, out, diag)
				}
				return r.encodeQ80MatvecQuantized(tensor, q, out)
			})

			// The artifact, built the way buildDenseArtifact builds it: one
			// aligned plane per diagonal block, laid end to end.
			plane := denseQ8AlignedBytes(perDiag, cols)
			if plane == 0 {
				t.Fatalf("denseQ8AlignedBytes(%d, %d) = 0", perDiag, cols)
			}
			aligned := cudaExpertAllocBytes(t, r, plane*diag)
			nblk := uint64(perDiag) * uint64(cols/32)
			for g := range diag {
				src := tensor.device + driver.DevicePtr(uint64(g)*nblk*34)
				if err := r.repackQ80(aligned.Slice(g*plane, (g+1)*plane), src, nblk); err != nil {
					t.Fatal(err)
				}
			}
			if err := syncStream(r.stream()); err != nil {
				t.Fatal(err)
			}

			got := run(func(out cudamem.Slice[float32]) error {
				if diag > 1 {
					return r.encodeQ80MatvecQuantizedBlockDiagAligned(aligned, uint64(cols), uint64(rows), q, out, diag)
				}
				return r.encodeQ80MatvecQuantizedAligned(aligned, uint64(cols), uint64(rows), q, out)
			})

			var nonzero int
			for _, v := range want {
				if v != 0 && !math.IsNaN(float64(v)) {
					nonzero++
				}
			}
			if nonzero < rows/2 {
				t.Fatalf("reference output is degenerate: only %d of %d rows are finite and nonzero", nonzero, rows)
			}
			if d := cudaQ81AlignedBitDiff(got, want); d >= 0 {
				t.Fatalf("aligned kernel differs from the raw-block kernel at row %d: %v (0x%08x) vs %v (0x%08x)",
					d, got[d], math.Float32bits(got[d]), want[d], math.Float32bits(want[d]))
			}

			// Positive controls: one byte of each plane of the LAST diagonal,
			// so a kernel that ignores the plane stride fails here too.
			base := (diag - 1) * plane
			dqBytes := int((nblk*2 + 63) &^ 63)
			for _, probe := range []struct {
				name   string
				offset int
			}{
				{"dq plane", base},
				{"qs plane", base + dqBytes},
			} {
				t.Run(probe.name, func(t *testing.T) {
					orig := make([]byte, 1)
					one := aligned.Slice(probe.offset, probe.offset+1)
					if err := one.CopyToHost(orig); err != nil {
						t.Fatal(err)
					}
					if err := one.CopyFromHost([]byte{orig[0] ^ 0x40}); err != nil {
						t.Fatal(err)
					}
					defer func() {
						if err := one.CopyFromHost(orig); err != nil {
							t.Fatal(err)
						}
					}()
					perturbed := run(func(out cudamem.Slice[float32]) error {
						if diag > 1 {
							return r.encodeQ80MatvecQuantizedBlockDiagAligned(aligned, uint64(cols), uint64(rows), q, out, diag)
						}
						return r.encodeQ80MatvecQuantizedAligned(aligned, uint64(cols), uint64(rows), q, out)
					})
					if cudaQ81AlignedBitDiff(perturbed, want) < 0 {
						t.Fatalf("flipping a byte of the %s changed nothing: the comparison does not read it", probe.name)
					}
				})
			}
		})
	}
}
