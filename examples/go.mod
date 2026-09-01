module github.com/tmc/lib/examples

go 1.27.0

require (
	github.com/tmc/apple v0.0.0
	github.com/tmc/lib/apple v0.0.0
	github.com/tmc/lib/nvidia v0.1.0
)

require (
	github.com/ebitengine/purego v0.11.0-alpha.11 // indirect
	golang.org/x/image v0.38.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/tmc/lib/nvidia => ../nvidia

replace github.com/tmc/lib/apple => ../apple

replace github.com/tmc/apple => ../../../tmc/apple
