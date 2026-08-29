package libp2p

import (
	"os"
	"testing"
)

func TestIntegrationWAN(t *testing.T) {
	if os.Getenv("LIBP2P_TEST_WAN") != "1" {
		t.Skip("set LIBP2P_TEST_WAN=1 and provide explicit WAN/bootstrap test configuration")
	}
	t.Fatal("manual WAN integration body is not generated")
}
