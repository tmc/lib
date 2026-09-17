package nvvm

// The option inventory is from nvvm.h (nvvmCompileProgram).
//go:generate go run github.com/tmc/lib/nvidia/x/internal/cudatool/cmd/genoptions -target=nvvm -out=flags.gen.go
