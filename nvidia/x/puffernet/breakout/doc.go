// Package breakout runs PufferLib 5.0's CUDA Breakout environment.
//
// breakout.cu is PufferLib's ocean/breakout/breakout.cu, unchanged. [New]
// compiles it with NVRTC and runs its reset and step kernels; the headers in
// shim stand in for the host headers it includes, whose code NVRTC does not
// compile. Observations are bfloat16, as in PufferLib.
//
// The environments step exactly as PufferLib's own nvcc build of
// breakout.cu does. testdata/cref.cu is that build, and TestTrajectory
// compares the two.
//
// breakout_opt.cu is a rewrite of breakout.cu with the same host API and
// outputs checked against the recorded trajectory, that [NewOpt] runs. It
// steps one environment per thread from state held in registers, instead of
// eight threads per environment from state in global memory. BenchmarkStep
// compares the two implementations at several batch sizes. Like
// breakout.cu, it never writes observations 8 and 9 (num_balls and
// paddle_width): breakout.cu writes its 10 scalar observations from lanes 0
// to 9 of an environment's 8 threads. The tests and benchmarks run both;
// build testdata/cref.cu with -DOPT to compare it in C.
//
// Usage:
//
//	v, err := breakout.New(dev, breakout.DefaultConfig, numEnvs)
//	...
//	err = v.Reset(stream, obs, rewards, terminals)
//	...
//	err = v.Step(stream, actions, obs, rewards, terminals)
package breakout
