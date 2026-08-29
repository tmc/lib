package libp2p

import "errors"

var (
	ErrClosed          = errors.New("libp2p: closed")
	ErrInvalidArgument = errors.New("libp2p: invalid argument")
)
