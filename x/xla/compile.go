package xla

// CompileOptions defines options for compiling an XLA program via PJRT.
type CompileOptions struct {
	ReplicaCount  int32
	NumPartitions int32
}

// DefaultCompileOptions returns standard defaults for compilation.
func DefaultCompileOptions() *CompileOptions {
	return &CompileOptions{
		ReplicaCount:  1,
		NumPartitions: 1,
	}
}

// ToProto serializes the CompileOptions to the equivalent CompileOptionsProto
// byte representation, using a hand-rolled protobuf encoder to adhere to the
// minimal external dependency constraint.
func (o *CompileOptions) ToProto() []byte {
	// CompileOptionsProto contains:
	// - Field 3: executable_build_options (ExecutableBuildOptionsProto)
	// Inside ExecutableBuildOptionsProto:
	// - Field 4: num_replicas (varint)
	// - Field 5: num_partitions (varint)
	var ebo []byte

	// ExecutableBuildOptionsProto fields
	if o.ReplicaCount > 0 {
		ebo = appendVarint(ebo, 4<<3|0)
		ebo = appendVarint(ebo, uint64(o.ReplicaCount))
	}
	if o.NumPartitions > 0 {
		ebo = appendVarint(ebo, 5<<3|0)
		ebo = appendVarint(ebo, uint64(o.NumPartitions))
	}

	var b []byte
	// Field 3: executable_build_options (message) inside CompileOptionsProto
	if len(ebo) > 0 {
		b = appendVarint(b, 3<<3|2) // Field 3, wire type 2 (length-delimited)
		b = appendVarint(b, uint64(len(ebo)))
		b = append(b, ebo...)
	}

	// Field 4: num_replicas (varint)
	if o.ReplicaCount > 0 {
		b = appendVarint(b, 4<<3|0)
		b = appendVarint(b, uint64(o.ReplicaCount))
	}

	// Field 5: num_partitions (varint)
	if o.NumPartitions > 0 {
		b = appendVarint(b, 5<<3|0)
		b = appendVarint(b, uint64(o.NumPartitions))
	}

	return b
}

func appendVarint(b []byte, v uint64) []byte {
	for v >= 1<<7 {
		b = append(b, uint8(v&0x7f|0x80))
		v >>= 7
	}
	b = append(b, uint8(v))
	return b
}

// Compile compiles a program for the given engine's client.
func (e *Engine) Compile(programCode []byte, format string, opts *CompileOptions) (*Executable, error) {
	if opts == nil {
		opts = DefaultCompileOptions()
	}
	optsBytes := opts.ToProto()
	if opts.ReplicaCount <= 1 && opts.NumPartitions <= 1 {
		optsBytes = defaultCompileOptionsProtoBytes
	}

	wrapper, err := e.client.Compile(programCode, format, optsBytes)
	if err != nil {
		return nil, err
	}

	return &Executable{
		engine:  e,
		wrapper: wrapper,
	}, nil
}
