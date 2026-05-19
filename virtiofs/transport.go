package virtiofs

import "context"

type Transport interface {
	RoundTrip(ctx context.Context, request []byte) ([]byte, error)
}
