//go:build linux && tpu

package main

import (
	"unsafe"

	"github.com/tmc/lib/openxla"
)

// newClientWrapperForTest constructs a ClientWrapper for unit tests bypassing unexported field restrictions.
func newClientWrapperForTest(api *openxla.Api, client *openxla.Client) *openxla.ClientWrapper {
	type clientWrapperInternal struct {
		api    *openxla.Api
		client *openxla.Client
	}
	cw := &clientWrapperInternal{
		api:    api,
		client: client,
	}
	return (*openxla.ClientWrapper)(unsafe.Pointer(cw))
}

// newExecutableWrapperForTest constructs an ExecutableWrapper for unit tests bypassing unexported field restrictions.
func newExecutableWrapperForTest(api *openxla.Api, exec *openxla.LoadedExecutable) *openxla.ExecutableWrapper {
	type executableWrapperInternal struct {
		api        *openxla.Api
		executable *openxla.LoadedExecutable
	}
	ew := &executableWrapperInternal{
		api:        api,
		executable: exec,
	}
	return (*openxla.ExecutableWrapper)(unsafe.Pointer(ew))
}
