// Package cudatmem models the checked device Tensor Memory allocation
// protocol for the sm_100a target family. It keeps hardware columns,
// logical slices, collective participation, and asynchronous lifetime
// separate. sm_121a is rejected because this package requires tcgen05 TMEM.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cudatmem
