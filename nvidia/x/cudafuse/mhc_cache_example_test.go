package cudafuse_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudacache"
	"github.com/tmc/lib/nvidia/x/cudafuse"
)

func ExampleCompileMHCCached() {
	config := cudafuse.MHCConfig{Batch: 1, Width: 32, Repeat: 4, Epsilon: 1e-6}
	_, err := cudafuse.CompileMHCCached(context.Background(), config, 12, 1, nil, cudacache.ToolProvenance{})
	fmt.Println(errors.Is(err, cudafuse.ErrMHCCompilerCacheUnavailable))
	// Output: true
}
