module github.com/tmc/lib/nvidia

go 1.27.0

require (
	github.com/aclements/go-moremath v0.0.0-20210112150236-f10218a38794
	github.com/ebitengine/purego v0.11.0
	github.com/google/pprof v0.0.0-20260906184651-6331bc6350fe
	github.com/ianlancetaylor/demangle v0.0.0-20250417193237-f615e6bd150b
	github.com/tmc/lib/egraphcore v0.0.0
	golang.org/x/perf v0.0.0-20260908200009-22c9c6c9d4da
	golang.org/x/sys v0.48.0
	golang.org/x/tools v0.50.0
	rsc.io/script v0.0.2
)

replace github.com/tmc/lib/egraphcore => ../egraphcore

require (
	github.com/tmc/lib/ebpfhost v0.0.0
	golang.org/x/mod v0.41.0 // indirect
	golang.org/x/sync v0.23.0 // indirect
)

replace github.com/tmc/lib/ebpfhost => ../ebpfhost
