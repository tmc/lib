# CUDA Tile IR bytecode goldens

Thirteen bytecode files written by NVIDIA's toolchain, copied verbatim from
the CUDA Tile IR project:

	repository: github.com/NVIDIA/cuda-tile
	commit:     7e8e2e68fa219716103824c01f7303367cf7df8d (2026-09-10)
	path:       test/Bytecode/encoding_stability/goldens
	license:    Apache License 2.0 with LLVM Exceptions (see the repository)

They are here because they are the only encoder evidence in this package that
we did not author. TestBytecodeGoldenReencode decodes each one and re-encodes
it, requiring byte-for-byte identity. Previously it read them through an
absolute path into a local checkout, so it skipped silently on every machine
but one -- including the GPU host -- and the strongest test in the package was
invisible to everyone else.

What they cover: the string (0x01), function (0x02), constant (0x04) and type
(0x05) sections. Nothing here exercises debug (0x03), global (0x06) or
producer (0x07). Identity on these files says we reproduce bytes NVIDIA
produced; it does not say NVIDIA's assembler accepts bytecode we author from
an AST. That is x/tileiras's TestAssembleNative.
