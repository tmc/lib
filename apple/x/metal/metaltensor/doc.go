// Package metaltensor builds symbolic Metal computation graphs.
//
// Compile currently lowers float32 elementwise Add, Mul, and SiLU expressions.
// Other operations may be represented in the graph but are rejected by Compile.
package metaltensor
