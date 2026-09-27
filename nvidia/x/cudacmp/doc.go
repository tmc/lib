// Package cudacmp implements pure-Go differential CUDA performance analysis.
//
// It ingests CUPTI activity captures and decomposes latency across inference
// phases (prefill and decode), normalizes kernel symbols into canonical
// architectural families, and evaluates differential speedups against
// statistical drift controls and physical roofline ceilings.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudacmp
