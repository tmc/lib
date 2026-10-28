module github.com/tmc/lib/kernelsearch

go 1.27.0

require github.com/tmc/lib/nvidia v0.0.0

require (
	github.com/ebitengine/purego v0.11.0 // indirect
	github.com/google/pprof v0.0.0-20260906184651-6331bc6350fe // indirect
	github.com/ianlancetaylor/demangle v0.0.0-20250417193237-f615e6bd150b // indirect
	github.com/tmc/lib/ebpfhost v0.0.0 // indirect
	github.com/tmc/lib/egraphcore v0.0.0 // indirect
	golang.org/x/mod v0.41.0 // indirect
	golang.org/x/sys v0.48.0 // indirect
)

replace github.com/tmc/lib/nvidia => ../nvidia

replace github.com/tmc/lib/egraphcore => ../egraphcore

replace github.com/tmc/lib/ebpfhost => ../ebpfhost
