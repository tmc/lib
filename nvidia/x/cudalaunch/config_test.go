package cudalaunch

import (
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudashape"
	"testing"
)

func TestConfigPreservesZeroSettings(t *testing.T) {
	stream := new(cuda.Stream)
	for _, tt := range []struct {
		name        string
		config      Config
		grid, block driver.Dim3
		shmem       uint32
		stream      *cuda.Stream
	}{
		{"zero preserves", Config{}, driver.Dims(2, 3, 4), driver.Dims(32, 1, 1), 1024, stream},
		{"dimensions replace", Config{Grid: cudashape.Grid3D{X: 9, Y: 1, Z: 1}, Block: driver.Dims(64, 1, 1)}, driver.Dims(9, 1, 1), driver.Dims(64, 1, 1), 1024, stream},
		{"shared memory replaces", Config{SharedMem: 2048}, driver.Dims(2, 3, 4), driver.Dims(32, 1, 1), 2048, stream},
	} {
		t.Run(tt.name, func(t *testing.T) {
			l := Kernel(1).Grid(2, 3, 4).Block(32, 1, 1).SharedMem(1024).On(stream)
			if got := l.Config(tt.config); got != l || l.grid != tt.grid || l.block != tt.block || l.shmem != tt.shmem || l.stream != tt.stream {
				t.Fatalf("config lost existing settings: %+v", l)
			}
		})
	}
}
