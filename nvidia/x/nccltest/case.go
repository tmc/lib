package nccltest

import (
	"fmt"
	"math"
	"time"
)

// Operation names a host collective or point-to-point pattern from nccl-tests.
type Operation string

const (
	AllReduce     Operation = "all_reduce"
	AllGather     Operation = "all_gather"
	Broadcast     Operation = "broadcast"
	ReduceScatter Operation = "reduce_scatter"
	Reduce        Operation = "reduce"
	AllToAll      Operation = "alltoall"
	AllToAllV     Operation = "alltoallv"
	Scatter       Operation = "scatter"
	Gather        Operation = "gather"
	SendRecv      Operation = "sendrecv"
	Hypercube     Operation = "hypercube"
)

// Reduction selects the arithmetic used by a reduction collective.
type Reduction string

const (
	Sum       Reduction = "sum"
	Product   Reduction = "prod"
	Minimum   Reduction = "min"
	Maximum   Reduction = "max"
	Average   Reduction = "avg"
	PreMulSum Reduction = "mulsum"
)

// Case describes one test. Bytes is the requested nccl-tests size,
// before the operation's per-rank 16-byte alignment. Root is a communicator rank.
// Zero DataType means Float32; zero Reduction means Sum. InPlace uses the
// operation's defined buffer offsets.
type Case struct {
	// Spread selects generated all-to-all-v traffic in [0, 1].
	// Nil uses the reference default of one. Other operations reject it.
	Spread    *float64
	Operation Operation
	DataType  DataType
	Bytes     int
	Root      int
	Reduction Reduction
	InPlace   bool
}

// Layout contains element counts and per-rank offsets for a collective.
// Counts are elements, not bytes. Offset fields are strides multiplied
// by rank for in-place execution.
type Layout struct {
	SendCount  int
	RecvCount  int
	ParamCount int
	SendOffset int
	RecvOffset int
}

// Layout returns the count/alignment rules used by the reference host tests.
// It rejects zero-work cases instead of accepting a vacuous correctness result.
func (c Case) Layout(ranks int) (Layout, error) {
	if ranks < 1 || ranks > 1024 {
		return Layout{}, fmt.Errorf("nccltest: ranks %d outside [1, 1024]", ranks)
	}
	if c.Spread != nil && (c.Operation != AllToAllV || math.IsNaN(*c.Spread) || math.IsInf(*c.Spread, 0) || *c.Spread < 0 || *c.Spread > 1) {
		return Layout{}, fmt.Errorf("nccltest: spread requires alltoallv and a finite value in [0, 1]")
	}
	size := c.DataType.Size()
	if size == 0 {
		return Layout{}, fmt.Errorf("nccltest: unknown datatype %q", c.DataType)
	}
	if c.Bytes < size {
		return Layout{}, fmt.Errorf("nccltest: size must contain at least one element")
	}
	if c.Root < 0 || c.Root >= ranks {
		return Layout{}, fmt.Errorf("nccltest: root %d outside communicator", c.Root)
	}
	switch c.Reduction {
	case "", Sum, Product, Minimum, Maximum, Average, PreMulSum:
	default:
		return Layout{}, fmt.Errorf("nccltest: unknown reduction %q", c.Reduction)
	}
	n := c.Bytes / size
	l := Layout{SendCount: n, RecvCount: n, ParamCount: n}
	base := (n / ranks) &^ (16/size - 1)
	switch c.Operation {
	case AllReduce, Broadcast, Reduce:
	case SendRecv:
		if c.InPlace {
			return Layout{}, fmt.Errorf("nccltest: sendrecv in-place is unsupported by the reference")
		}
	case AllGather, Gather, Hypercube:
		l = Layout{SendCount: base, RecvCount: base * ranks, ParamCount: base, SendOffset: base}
	case ReduceScatter, Scatter:
		l = Layout{SendCount: base * ranks, RecvCount: base, ParamCount: base, RecvOffset: base}
	case AllToAll:
		l = Layout{SendCount: base * ranks, RecvCount: base * ranks, ParamCount: base}
		if c.InPlace {
			return Layout{}, fmt.Errorf("nccltest: alltoall in-place is unsupported by the reference")
		}
	case AllToAllV:
		if c.InPlace {
			return Layout{}, fmt.Errorf("nccltest: alltoallv in-place is unsupported by the reference")
		}
		total := 0
		for peer := 0; peer < ranks; peer++ {
			count := c.peerElements(base, ranks, 0, peer, size)
			if count < 0 || count > n-total {
				return Layout{}, fmt.Errorf("nccltest: alltoallv counts exceed requested capacity")
			}
			total += count
		}
		l = Layout{SendCount: total, RecvCount: total, ParamCount: base}
	default:
		return Layout{}, fmt.Errorf("nccltest: unknown operation %q", c.Operation)
	}
	if c.Operation == Hypercube && ranks&(ranks-1) != 0 {
		return Layout{}, fmt.Errorf("nccltest: hypercube requires a power-of-two rank count")
	}
	if l.ParamCount == 0 {
		return Layout{}, fmt.Errorf("nccltest: size rounds to zero work for %d ranks", ranks)
	}
	return l, nil
}

// Bandwidth returns algorithm and bus throughput in decimal GB/s. It follows
// each reference operation's byte convention; one-rank bus throughput is zero
// for collectives with no inter-rank traffic.
func (c Case) Bandwidth(ranks int, elapsed time.Duration) (algorithm, bus float64, err error) {
	l, err := c.Layout(ranks)
	if err != nil {
		return 0, 0, err
	}
	if elapsed <= 0 {
		return 0, 0, fmt.Errorf("nccltest: nonpositive elapsed time")
	}
	bytes := float64(l.ParamCount * c.DataType.Size())
	factor := 1.0
	switch c.Operation {
	case AllReduce:
		factor = 2 * float64(ranks-1) / float64(ranks)
	case AllGather, ReduceScatter, Gather, Scatter, AllToAll:
		bytes *= float64(ranks)
		factor = float64(ranks-1) / float64(ranks)
	case AllToAllV:
		bytes = float64(l.SendCount * c.DataType.Size())
		factor = float64(ranks-1) / float64(ranks)
	case Hypercube:
		bytes *= float64(ranks - 1)
	}
	algorithm = bytes / elapsed.Seconds() / 1e9
	return algorithm, algorithm * factor, nil
}

// Sizes generates an inclusive geometric byte sweep with overflow checks.
func Sizes(first, last, factor int) ([]int, error) {
	if first <= 0 || last < first || factor < 2 {
		return nil, fmt.Errorf("nccltest: invalid size sweep")
	}
	var sizes []int
	for n := first; n <= last; {
		sizes = append(sizes, n)
		if n > last/factor {
			break
		}
		n *= factor
	}
	return sizes, nil
}

// LinearSizes generates an inclusive byte sweep with a positive increment.
// The last size is included only when it lies on the sweep.
func LinearSizes(first, last, increment int) ([]int, error) {
	if first <= 0 || last < first || increment <= 0 {
		return nil, fmt.Errorf("nccltest: invalid linear size sweep")
	}
	var sizes []int
	for n := first; ; n += increment {
		sizes = append(sizes, n)
		if increment > last-n {
			return sizes, nil
		}
	}
}

// copy snapshots caller-owned optional controls, including in yielded results.
func (c Case) copy() Case {
	if c.Spread != nil {
		spread := *c.Spread
		c.Spread = &spread
	}
	return c
}
