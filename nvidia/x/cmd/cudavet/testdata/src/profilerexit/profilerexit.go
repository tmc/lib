package main

import (
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/tmc/lib/nvidia/driver"
)

// device is shaped like cuda.Device, whose WithContext runs fn with the
// device's context current.
type device struct{}

func (d *device) WithContext(fn func() error) error { return fn() }

func work() error { return errors.New("failed") }

func fatal(dev *device) {
	defer dev.WithContext(driver.ProfilerStop)
	if err := work(); err != nil {
		log.Fatal(err) // want `log.Fatal skips the deferred driver.ProfilerStop and an attached profiler is never flushed; return instead, or call ProfilerStop first`
	}
}

func literal() {
	defer func() { driver.ProfilerStop() }()
	if err := work(); err != nil {
		os.Exit(1) // want `os.Exit skips the deferred driver.ProfilerStop`
	}
}

func stopFirst(dev *device) {
	defer dev.WithContext(driver.ProfilerStop)
	if err := work(); err != nil {
		dev.WithContext(driver.ProfilerStop)
		os.Exit(1)
	}
}

func exitBeforeDefer(dev *device, bad bool) {
	if bad {
		os.Exit(2)
	}
	defer dev.WithContext(driver.ProfilerStop)
	work()
}

func panics(dev *device) {
	defer dev.WithContext(driver.ProfilerStop)
	if err := work(); err != nil {
		panic(err)
	}
}

func run(dev *device) int {
	defer dev.WithContext(driver.ProfilerStop)
	if err := work(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	return 0
}

func exitsRun() {
	os.Exit(run(new(device)))
}

func main() {
	dev := new(device)
	defer dev.WithContext(driver.ProfilerStop)
	if err := work(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
}
