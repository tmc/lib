package libp2p_test

import (
	"fmt"

	"github.com/tmc/lib/libp2p"
)

func Example() {
	cfg := libp2p.DefaultConfig()
	fmt.Println(cfg.ListenAddrs[0])
	// Output:
	// /ip4/127.0.0.1/tcp/0
}
