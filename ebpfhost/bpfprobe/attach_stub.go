//go:build !linux || (!amd64 && !arm64)

package bpfprobe

import (
	"github.com/tmc/lib/ebpfhost/bpfload"
	"os"
)

func AttachGroup(configs []Config) (*Group, error) {
	for _, cfg := range configs {
		if _, err := validateConfig(cfg); err != nil {
			return nil, err
		}
	}
	return nil, bpfload.ErrUnsupported
}
func disablePerf(*os.File) error { return bpfload.ErrUnsupported }

// Attach loads and attaches an upstream Linux BPF uprobe on amd64.
func Attach(c Config) (*Probe, error) {
	if _, err := validateConfig(c); err != nil {
		return nil, err
	}
	return nil, bpfload.ErrUnsupported
}
