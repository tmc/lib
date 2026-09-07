module github.com/tmc/lib/examples

go 1.27.0

require (
	github.com/tmc/apple v0.0.0
	github.com/tmc/go-iroh v0.0.0-00010101000000-000000000000
	github.com/tmc/lib/apple v0.0.0
	github.com/tmc/lib/hfsync v0.0.0
	github.com/tmc/lib/lm v0.0.0
	github.com/tmc/lib/nvidia v0.1.0
	github.com/tmc/lib/openxla v0.0.0-00010101000000-000000000000
	github.com/tmc/lib/tpu v0.0.0-00010101000000-000000000000
	github.com/tmc/modelir v0.0.0
)

require (
	filippo.io/edwards25519 v1.2.0 // indirect
	github.com/coder/websocket v1.8.14 // indirect
	github.com/ebitengine/purego v0.11.0 // indirect
	github.com/klauspost/cpuid/v2 v2.0.9 // indirect
	github.com/quic-go/quic-go v0.62.0 // indirect
	golang.org/x/crypto v0.54.0 // indirect
	golang.org/x/net v0.56.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	lukechampine.com/blake3 v1.4.1 // indirect
)

replace github.com/tmc/lib/hfsync => ../hfsync

replace github.com/tmc/lib/lm => ../lm

replace github.com/tmc/lib/nvidia => ../nvidia

replace github.com/tmc/lib/apple => ../apple

replace github.com/tmc/apple => ../../../tmc/apple

replace github.com/tmc/modelir => ../../modelir

replace github.com/tmc/go-iroh => ../../go-iroh

replace github.com/tmc/lib/openxla => ../openxla

replace github.com/tmc/lib/tpu => ../tpu
