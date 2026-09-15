// Package gguf reads GGUF model containers.
//
// GGUF is the single-file container used by llama.cpp and compatible runtimes.
// A file holds a header, a metadata key-value block, a tensor directory, and
// the tensor data itself. [Read] parses everything but the tensor data, which
// callers map or read themselves using [File.DataOffset], [Tensor.Offset], and
// [Tensor.Bytes].
//
// Read is strict: it accepts only version 3 files and rejects duplicate keys,
// non-UTF-8 strings, unaligned or out-of-order tensor offsets, and any declared
// size that does not fit the file. Parsing is bounded, so a truncated or
// hostile file yields an error rather than a large allocation. Errors from a
// malformed file wrap [ErrInvalid]; errors from a well-formed file that uses an
// encoding this package does not implement wrap [ErrUnsupported].
//
//	g, err := gguf.Read(f, size)
//	if err != nil {
//		return err
//	}
//	for _, t := range g.Tensors {
//		n, err := t.Bytes() // tensor occupies [g.DataOffset+t.Offset, +n)
//		...
//	}
package gguf
