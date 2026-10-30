package nvidiasmi

import (
	"bytes"
	"context"
)

// Query runs nvidia-smi -q -x, found in $PATH, and parses its output.
// If nvidia-smi exits with a nonzero status, the error is an *[Error].
func Query(ctx context.Context) (*Log, error) {
	out, err := run(ctx, "-q", "-x")
	if err != nil {
		return nil, err
	}
	return Parse(bytes.NewReader(out))
}
