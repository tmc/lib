package httpapi

import (
	"errors"
	"net"
	"strings"
)

// Listen listens on unix:/absolute/socket, or tcp:host:port. An empty address
// selects unix:/var/run/cudahotswap.sock. Existing socket paths are not removed.
// The embedding application controls directory permissions and shutdown.
func Listen(address string) (net.Listener, error) {
	if address == "" {
		address = "unix:/var/run/cudahotswap.sock"
	}
	if strings.HasPrefix(address, "unix:") {
		path := strings.TrimPrefix(address, "unix:")
		if !strings.HasPrefix(path, "/") {
			return nil, errors.New("unix socket must be absolute")
		}
		return net.Listen("unix", path)
	}
	if strings.HasPrefix(address, "tcp:") {
		peer := strings.TrimPrefix(address, "tcp:")
		if _, _, err := net.SplitHostPort(peer); err != nil {
			return nil, err
		}
		return net.Listen("tcp", peer)
	}
	return nil, errors.New("address must be unix:path or tcp:host:port")
}
