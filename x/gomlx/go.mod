module github.com/tmc/lib/x/gomlx

go 1.27.0

require (
	github.com/gomlx/compute v0.1.8
	github.com/gomlx/gomlx v0.28.11-0.20260906081732-2ede728a8eb8
	github.com/tmc/lib/x/stablehlo v0.0.0
	github.com/tmc/lib/x/xla v0.0.0
)

require (
	github.com/ebitengine/purego v0.11.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/tmc/lib/openxla v0.0.0-00010101000000-000000000000 // indirect
	golang.org/x/exp v0.0.0-20260410095643-746e56fc9e2f // indirect
	k8s.io/klog/v2 v2.140.0 // indirect
)

replace github.com/tmc/lib/x/xla => ../xla

replace github.com/tmc/lib/x/stablehlo => ../stablehlo

replace github.com/tmc/lib/openxla => ../../openxla
