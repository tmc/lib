// Package cudatmem models the checked device Tensor Memory allocation
// protocol for the sm_100a target family. It keeps hardware columns,
// logical slices, collective participation, and asynchronous lifetime
// separate. sm_121a is rejected because this package requires tcgen05 TMEM.
package cudatmem
