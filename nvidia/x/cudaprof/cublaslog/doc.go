// Package cublaslog parses the lines cuBLASLt's logger writes for matmul
// calls.
//
// At logger mask 2 ("performance trace") cuBLASLt describes each matmul it
// runs in one line of key=value fields, with the matrix layouts, the compute
// descriptor and the algorithm as bracketed groups:
//
//	A=0X32F600000 Adesc=[type=R_16BF rows=1024 cols=768 ld=1024] B=... Bdesc=[...]
//	C=... Cdesc=[...] D=... Ddesc=[...] computeDesc=[computeType=COMPUTE_32F
//	scaleType=R_32F transa=OP_T] algo=[algoId=21 tile=MATMUL_TILE_16x16
//	stages=MATMUL_STAGES_32x1] workSpace=... beta=0 outOfPlace=0 stream=0X0
//
// The library omits a field whose value is the default, so [Parse] fills in
// the defaults cublasLt.h documents. The header calls the logger
// experimental; Parse keeps every field it saw in [Matmul.Fields], so a
// field it does not know is still available.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch releases.
package cublaslog
