// Package cudabench provides precise GPU event-timed benchmarking utilities for Go testing.
//
// Unlike standard CPU wall-clock benchmarks which measure host dispatch latency and queueing,
// cudabench uses hardware [cuda.Event] timestamps to record kernel execution time directly on
// the device SMs, reporting standard metrics including gpu-ns/op, throughput (decimal GB/s,
// where 1 GB is 1e9 bytes), and compute (TFLOPS).
//
// # Usage
//
//	func BenchmarkVectorAdd(b *testing.B) {
//		dev := cudatest.Require(b)
//		cudabench.Run(b, dev, func(cb *cudabench.B) {
//			cb.SetBytes(int64(n * 4 * 3))
//			cb.Time(func(s *cuda.Stream) {
//				_ = launcher.On(s).Launch(...)
//			})
//		})
//	}
package cudabench
