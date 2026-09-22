// The primary file of a two-file translation unit, as a compilation of a file
// that includes others produces: one CUBIN and one PTX, whose line directives
// name every file that took part. It defines no kernel itself, so resolving
// the unit's entry point requires searching past it.
#include "impl.cu"
