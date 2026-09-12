package nccltest

import (
	"fmt"
	"math"
)

// input varies with rank and position, using small integers exactly
// representable as float32.
func input(rank, index int) float32 {
	x := uint32(rank+1)*0x9e3779b9 + uint32(index)
	x ^= x >> 16
	x *= 0x85ebca6b
	x ^= x >> 13
	x *= 0xc2b2ae35
	x ^= x >> 16
	return float32(1 + x%13)
}

type floating interface{ ~float32 | ~float64 }

// Double inputs include bits below float32 precision so an accidental narrow
// transfer or arithmetic path cannot pass a float64 correctness check.
func inputValue[T floating](c Case, rank, index int) T {
	value := T(input(rank, index))
	// Four magnitude-bearing factors keep products below 14^4. Later
	// ranks still contribute independently varying signs, whose multiplication
	// is exact. This avoids overflow for every supported communicator size.
	if c.Reduction == Product && rank >= 4 {
		if int(value)%2 != 0 {
			return -1
		}
		return 1
	}
	if c.DataType == Float64 {
		value += T(float64(1+index%17+rank%7) / (1 << 32))
	}
	return value
}

func sourceElement(c Case, l Layout, ranks, rank, index int) (source, element int, isCopy bool) {
	switch c.Operation {
	case AllGather, Gather, Hypercube:
		return index / l.ParamCount, index % l.ParamCount, true
	case AllToAll:
		return index / l.ParamCount, rank*l.ParamCount + index%l.ParamCount, true
	case AllToAllV:
		offset := 0
		for peer := 0; peer < ranks; peer++ {
			count := c.peerElements(l.ParamCount, ranks, peer, rank, c.DataType.Size())
			if index < offset+count {
				sourceOffset := 0
				for dest := 0; dest < rank; dest++ {
					sourceOffset += c.peerElements(l.ParamCount, ranks, peer, dest, c.DataType.Size())
				}
				return peer, sourceOffset + index - offset, true
			}
			offset += count
		}
		return 0, 0, true
	case Scatter:
		return c.Root, rank*l.ParamCount + index, true
	case Broadcast:
		return c.Root, index, true
	case SendRecv:
		return (rank + ranks - 1) % ranks, index, true
	}
	return 0, 0, false
}

func expectedFloat[T floating](c Case, l Layout, ranks, rank, index int) T {
	if source, element, isCopy := sourceElement(c, l, ranks, rank, index); isCopy {
		return inputValue[T](c, source, element)
	}
	if c.Operation == ReduceScatter {
		index += rank * l.ParamCount
	}
	value := inputValue[T](c, 0, index)
	if c.Reduction == PreMulSum {
		value *= T(premulScalar(0))
	}
	for r := 1; r < ranks; r++ {
		x := inputValue[T](c, r, index)
		switch c.Reduction {
		case PreMulSum:
			value += x * T(premulScalar(r))
		case "", Sum, Average:
			value += x
		case Product:
			value *= x
		case Minimum:
			if x < value {
				value = x
			}
		case Maximum:
			if x > value {
				value = x
			}
		}
	}
	if c.Reduction == Average {
		value /= T(ranks)
	}
	return value
}

// Check compares a rank's Float32 values against an independent host oracle.
// Non-root reduce/gather ranks have no defined receive output and return zero
// comparisons. Run must still require a positive total across the communicator.
// Nonfinite expected values return an error instead of an infinite tolerance.
func Check(c Case, ranks, rank int, got []float32) (compared, wrong int, err error) {
	if c.DataType != "" && c.DataType != Float32 {
		return 0, 0, fmt.Errorf("nccltest: Check requires float32")
	}
	return checkFloat(c, ranks, rank, got, 1e-5)
}

// Check64 compares Float64 output without converting it to float32.
// It follows Check's rules for non-root outputs and comparison counts.
func Check64(c Case, ranks, rank int, got []float64) (compared, wrong int, err error) {
	if c.DataType != Float64 {
		return 0, 0, fmt.Errorf("nccltest: Check64 requires float64")
	}
	return checkFloat(c, ranks, rank, got, 1e-12)
}

func checkFloat[T floating](c Case, ranks, rank int, got []T, relativeTolerance float64) (compared, wrong int, err error) {
	l, err := c.Layout(ranks)
	if err != nil {
		return 0, 0, err
	}
	if rank < 0 || rank >= ranks {
		return 0, 0, fmt.Errorf("nccltest: invalid rank %d", rank)
	}
	if (c.Operation == Reduce || c.Operation == Gather) && rank != c.Root {
		return 0, 0, nil
	}
	if len(got) != l.RecvCount {
		return 0, 0, fmt.Errorf("nccltest: received %d elements, want %d", len(got), l.RecvCount)
	}
	for i, value := range got {
		want := expectedFloat[T](c, l, ranks, rank, i)
		if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
			return i, wrong, fmt.Errorf("nccltest: nonfinite expected value at rank %d element %d", rank, i)
		}
		tolerance := relativeTolerance * math.Max(1, math.Abs(float64(want)))
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-want)) > tolerance {
			wrong++
		}
	}
	return len(got), wrong, nil
}

// Default upstream alltoallv spread is 1: distance weighted and 16-byte aligned.
func peerElements(base, ranks, source, destination, size int) int {
	return (Case{}).peerElements(base, ranks, source, destination, size)
}

func (c Case) peerElements(base, ranks, source, destination, size int) int {
	if ranks == 1 {
		return base
	}
	distance := (destination - source + ranks) % ranks
	total := float64(base * ranks * size)
	spread := 1.0
	if c.Spread != nil {
		spread = *c.Spread
	}
	uniform := total / float64(ranks)
	weighted := float64(distance) * total / float64(ranks*(ranks-1)/2)
	bytes := (1-spread)*uniform + spread*weighted
	// Out-of-range float-to-int conversions are implementation-dependent.
	// Reject before conversion; Layout reports the invalid peer count.
	if bytes >= float64(int(^uint(0)>>1)) {
		return -1
	}
	return (int(bytes) &^ 15) / size
}

// nccl-tests/verifiable/verifiable.h alternates 1 and 2 by communicator rank.
func premulScalar(rank int) float32 { return float32(1 + rank%2) }
