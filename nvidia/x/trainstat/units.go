package trainstat

import (
	"runtime"
	"runtime/debug"
)

// Units returns the standard unit metadata for the names this package
// knows: training losses, evaluation metrics, step times and run status.
// Each call returns a new map.
func Units() map[string][]string {
	return map[string][]string{
		"loss":      {"better=lower", "assume=normal"},
		"ce":        {"better=lower", "assume=normal"},
		"rps":       {"better=lower", "assume=normal"},
		"act":       {"better=lower", "assume=normal"},
		"mae":       {"better=lower", "assume=normal"},
		"grad-norm": {"assume=normal"},
		"accuracy":  {"better=higher", "assume=normal", "delta=abs"},
		"act-auroc": {"better=higher", "assume=normal", "delta=abs"},
		"spearman":  {"better=higher", "assume=normal", "delta=abs"},
		"brier":     {"better=lower", "assume=normal", "delta=abs"},
		"ece":       {"better=lower", "assume=normal", "delta=abs"},
		"aurc":      {"better=lower", "assume=normal", "delta=abs"},
		"sec/step":  {"better=lower", "kind=time"},
		"tokens/s":  {"better=higher", "kind=time"},
		"train-sec": {"better=lower", "kind=time"},
		"done":      {"better=higher", "delta=abs"},
		"nonfinite": {"better=lower", "delta=abs"},
	}
}

// Build returns the configuration that describes the running binary:
// commit (the VCS revision, with a -dirty suffix if the tree was modified)
// and go (the toolchain version). Commit is omitted when the binary
// records no revision, as test binaries do not. Versions of libraries the
// run depends on, such as lib-nvidia or cublas, are the caller's to
// record; a replaced module has no version and is recorded as "(devel)",
// not as an invented commit.
func Build() [][2]string {
	var kv [][2]string
	if bi, ok := debug.ReadBuildInfo(); ok {
		var rev string
		var dirty bool
		for _, s := range bi.Settings {
			switch s.Key {
			case "vcs.revision":
				rev = s.Value
			case "vcs.modified":
				dirty = s.Value == "true"
			}
		}
		if rev != "" {
			if dirty {
				rev += "-dirty"
			}
			kv = append(kv, [2]string{"commit", rev})
		}
	}
	return append(kv, [2]string{"go", runtime.Version()})
}
