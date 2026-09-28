// Package puffernet runs the default PufferLib 5.0 policy network on CUDA.
//
// The network is a bias-free linear encoder, a stack of MinGRU layers with
// sigmoid highway connections, and a linear decoder whose last output is the
// value estimate. It is the Arch built by build_arch in PufferLib's
// src/algo.cu, with the inference step of src/puffercpu.c.
//
// [LoadWeights] reads a checkpoint written by ./puffer train. [Net] runs one
// recurrent step per call for a batch of agents, one thread block per agent.
// [Host] is the same step in Go, following puffercpu.c operation for
// operation; the tests use it as the reference.
//
// Usage, for a checkpoint of the breakout environment:
//
//	cfg := puffernet.Config{InputSize: 118, HiddenSize: 32, NumLayers: 4, ActionSizes: []int{3}}
//	w, err := puffernet.LoadWeights("checkpoints/breakout/.../0000000054919168.bin", cfg)
//	...
//	net, err := puffernet.New(dev, w, numAgents)
//	...
//	err = net.Step(stream, out, obs, terminals)
//
// Each row of out holds the logits of every action head, concatenated, and
// then the value. For continuous actions the logits are the action means and
// [Weights].LogStd holds their log standard deviations.
package puffernet
