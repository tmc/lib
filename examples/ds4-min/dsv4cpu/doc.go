// Package dsv4cpu provides native-Go DeepSeek V4 CPU tensor operations.
//
// Matrix reads immutable serialized F32, F16, Q8_0, Q2_K, Q4_K and IQ2_XXS weights
// without expanding an entire model. RMSNorm and the hyperconnection operations
// follow upstream scalar reduction order.
// These operations are not yet a complete model backend.
//
// To regenerate the decoding tables, set DS4_CPU_ORACLE_SOURCE to the pinned
// checkout's ds4.c, then run go generate. The source hash is checked first.
package dsv4cpu

// The source argument locates a read-only pinned oracle checkout. The generator
// verifies the exact source hash before extracting either table.
//go:generate go run ./internal/gentables -out iq2_tables.go
