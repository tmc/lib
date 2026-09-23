package main

import "fmt"

type projectionLayout uint8

const (
	projectionRowMajor projectionLayout = iota + 1
)

type projectionPrecision uint8

const (
	projectionF32 projectionPrecision = iota + 1
	projectionF16
)

type projectionRoute struct {
	Layout      projectionLayout
	Precision   projectionPrecision
	Input       int
	Output      int
	Accumulator string
	Reduction   string
	Quant       string
}

type projectionGroup uint8

const (
	projectionQKV projectionGroup = iota + 1
	projectionGateUp
)

func eligibleProjectionFusion(group projectionGroup, routes ...projectionRoute) error {
	want := 3
	if group == projectionGateUp {
		want = 2
	}
	if group != projectionQKV && group != projectionGateUp {
		return fmt.Errorf("projection fusion: unknown group %d", group)
	}
	if len(routes) != want {
		return fmt.Errorf("projection fusion: group %d has %d routes, want %d", group, len(routes), want)
	}
	first := routes[0]
	if first.Layout != projectionRowMajor || first.Input <= 0 || first.Output <= 0 ||
		(first.Precision != projectionF32 && first.Precision != projectionF16) ||
		first.Accumulator != "f32" || first.Reduction != "strided-simd128" || first.Quant != "none" {
		return fmt.Errorf("projection fusion: route 0 is unsupported")
	}
	for i, route := range routes[1:] {
		if route.Layout != first.Layout || route.Precision != first.Precision ||
			route.Input != first.Input || route.Accumulator != first.Accumulator ||
			route.Reduction != first.Reduction || route.Quant != first.Quant || route.Output <= 0 {
			return fmt.Errorf("projection fusion: route %d differs from route 0", i+1)
		}
	}
	if group == projectionQKV && routes[1].Output != routes[2].Output {
		return fmt.Errorf("projection fusion: key and value widths differ")
	}
	if group == projectionGateUp && routes[0].Output != routes[1].Output {
		return fmt.Errorf("projection fusion: gate and up widths differ")
	}
	return nil
}

// useProjectionFusion reports whether a fused projection kernel may replace
// the supplied unfused routes. A false result is the complete fallback
// contract: callers must dispatch the original routes unchanged.
func useProjectionFusion(group projectionGroup, routes ...projectionRoute) bool {
	return eligibleProjectionFusion(group, routes...) == nil
}

func modelProjectionRoutes(cfg Config, half bool, group projectionGroup) []projectionRoute {
	precision := projectionF32
	if half {
		precision = projectionF16
	}
	base := projectionRoute{Layout: projectionRowMajor, Precision: precision, Input: cfg.Dim, Accumulator: "f32", Reduction: "strided-simd128", Quant: "none"}
	switch group {
	case projectionQKV:
		q, kv := base, base
		q.Output = cfg.NumHeads * cfg.HeadDim
		kv.Output = cfg.NumKVHeads * cfg.HeadDim
		return []projectionRoute{q, kv, kv}
	case projectionGateUp:
		base.Output = cfg.HiddenDim
		return []projectionRoute{base, base}
	default:
		return nil
	}
}
