package kvm_test

import (
	"fmt"

	"github.com/tmc/lib/kvm"
)

func ExampleOpen() {
	dev, err := kvm.Open()
	if err != nil {
		return
	}
	defer dev.Close()

	version, err := dev.APIVersion()
	if err != nil {
		return
	}
	fmt.Println(version == kvm.APIVersion)
}
