package main

import (
	"errors"
	"flag"

	"github.com/tmc/lib/examples/ds4-min/dsv4expertcache"
)

type streamingOptions struct {
	enabled   bool
	cold      bool
	autoCache bool
	budget    string
	preload   uint
}

func addStreamingFlags(fs *flag.FlagSet, o *streamingOptions) {
	fs.BoolVar(&o.enabled, "ssd-streaming", false, "load routed experts through a bounded raw cache")
	fs.BoolVar(&o.cold, "ssd-streaming-cold", false, "skip expert hotlist preload")
	fs.BoolVar(&o.autoCache, "ssd-streaming-auto-cache", false, "use the Go automatic cache budget extension")
	fs.StringVar(&o.budget, "ssd-streaming-cache-experts", "", "expert cache slot count or GiB budget (N or NGB)")
	fs.UintVar(&o.preload, "ssd-streaming-preload-experts", 0, "number of experts to preload")
}

func (o streamingOptions) validate(backend string) error {
	if !o.enabled {
		if o.cold || o.autoCache || o.budget != "" || o.preload != 0 {
			return errors.New("ssd-streaming options require ssd-streaming")
		}
		return nil
	}
	if backend == "cuda" {
		if o.autoCache || o.budget == "" || o.preload != 0 {
			return errors.New("ssd-streaming CUDA requires explicit budget and no preload")
		}
	}
	if backend != "metal" && backend != "cuda" {
		return errors.New("ssd-streaming backend is not qualified")
	}
	if o.autoCache && o.budget != "" {
		return errors.New("ssd-streaming auto cache conflicts with explicit budget")
	}
	if o.budget != "" {
		if _, err := dsv4expertcache.ParseBudget(o.budget); err != nil {
			return err
		}
	}
	if o.preload > uint(^uint32(0)) {
		return errors.New("ssd-streaming preload count overflows")
	}
	return nil
}
