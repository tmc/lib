// Package testvec is vec.cu, bound by cudabind. It tests cudabind.
package testvec

//go:generate go run ../.. -type vecModule -shared sum -std c++17 vec.cu
