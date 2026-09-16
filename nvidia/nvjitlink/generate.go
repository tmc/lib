package nvjitlink

// The option inventory is from https://docs.nvidia.com/cuda/nvjitlink/#supported-link-options.
//go:generate go run github.com/tmc/lib/nvidia/x/internal/cudatool/cmd/genoptions -target=nvjitlink -out=flags.gen.go
