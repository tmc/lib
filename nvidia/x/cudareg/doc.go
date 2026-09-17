// Package cudareg checks a bounded setmaxnreg producer-consumer protocol.
//
// It models the CTA register pool and warpgroup participation without
// pretending to perform compiler register allocation. Generated fixtures are
// PTX-backed source descriptions; compilation and device execution are
// separate, explicitly gated steps.
package cudareg
