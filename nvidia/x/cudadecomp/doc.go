// Package cudadecomp prepares weight tensors for the CUDA hardware
// decompression engine and stages them from storage through decompression
// to their consumer.
//
// [Compress] cuts a byte stream into independently compressed chunks, each
// tagged with a CRC-32C of its uncompressed bytes; [CompressBF16] first
// splits BF16 weights into a plane of high bytes (sign and seven exponent
// bits) and a plane of low bytes, which compresses better than the
// interleaved form because the exponents of trained weights cluster.
// [Tensor.Decompress] and [DecompressBF16] are the host reference decoders:
// a chunk whose bytes or checksum were altered yields an error wrapping
// [ErrChunkCorrupted], never wrong weights. The package carries its own LZ4
// block encoder and decoder; Deflate uses compress/flate.
//
// [Tensor.BatchParams] builds the parameter array of
// cuMemBatchDecompressAsync ([driver.Stream.MemBatchDecompressAsync]) for
// tensors placed at given device addresses, and [Tensor.ReadOps] lists the
// file reads a cuFile batch would issue to stage the payload. [Schedule]
// and [VerifySchedule] describe a three-stage read, decompress, consume
// pipeline over double buffers, and [Run] executes that schedule on the
// host.
//
// [EmitBytePlaneMergePTX] emits a PTX kernel that rebuilds bf16x2 words
// from the two planes with prmt.b32. Its tests interpret the emitted
// instructions in Go; with CUDA_DECOMP_NATIVE=1 they also assemble it with
// ptxas and report the device's decompression capabilities. No batch has
// been submitted to a hardware decompression engine by this package's
// authors, and throughput is not measured here.
//
// Snappy and ANS entropy coding are not implemented.
//
// This package is outside the compatibility promise of
// github.com/tmc/lib/nvidia. Its API may change or be removed in patch
// releases.
package cudadecomp
