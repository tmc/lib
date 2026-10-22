package main

import (
	"io"

	"github.com/tmc/lib/examples/ds4-min/internal/dsparkstate"
)

func writeDSparkSupportContainer(dst io.Writer, core []byte, support *dsparkSupportEnvelope, frontier int, generation uint64, maximum uint64) error {
	return dsparkstate.WriteContainer(dst, core, support.wire(), frontier, generation, maximum)
}

func readDSparkSupportContainer(src io.Reader, cuda bool, maximum uint64) (core, support []byte, generation uint64, err error) {
	kind := uint8(2)
	if cuda {
		kind = 1
	}
	return dsparkstate.ReadContainer(src, kind, maximum)
}
