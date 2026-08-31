module github.com/tmc/lib/examples

go 1.26.2

require github.com/tmc/lib/nvidia v0.1.0

require (
	github.com/ebitengine/purego v0.11.0-alpha.11 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/tmc/lib/nvidia => ../nvidia
