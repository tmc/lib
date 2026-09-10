module github.com/tmc/lib/x/xla

go 1.27.0

replace github.com/tmc/lib/openxla => ../../openxla

require (
	github.com/ebitengine/purego v0.11.0
	github.com/tmc/lib/openxla v0.0.0-00010101000000-000000000000
	github.com/tmc/lib/x/stablehlo v0.0.0-00010101000000-000000000000
)

require github.com/tmc/ffi v0.0.0-20260906092449-1be2d95a8044 // indirect

replace github.com/tmc/lib/x/stablehlo => ../stablehlo
