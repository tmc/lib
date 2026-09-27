// Package cudareg checks a bounded setmaxnreg producer-consumer protocol.
//
// It models the CTA register pool and warpgroup participation without
// pretending to perform compiler register allocation. Generated fixtures are
// PTX-backed source descriptions; compilation and device execution are
// separate, explicitly gated steps.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudareg
