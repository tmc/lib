// Package cudainject contains the source assets used by the injection agent.
package cudainject

import "embed"

// Sources contains the unchanged inputs compiled by native/build_agent.sh.
//
//go:embed native/cupti_agent.c native/cupti_memory.h native/cupti_dependency.h
var Sources embed.FS
