//go:build linux

package nbd

import "testing"

func TestKernelIOCTLRequests(t *testing.T) {
	for name, tc := range map[string]struct{ got, want uintptr }{
		"NBD_SET_SOCK":        {requestSetSock, 0xab00},
		"NBD_SET_BLKSIZE":     {requestSetBlockSize, 0xab01},
		"NBD_SET_SIZE":        {requestSetSize, 0xab02},
		"NBD_DO_IT":           {requestDoIt, 0xab03},
		"NBD_CLEAR_SOCK":      {requestClearSock, 0xab04},
		"NBD_CLEAR_QUE":       {requestClearQueue, 0xab05},
		"NBD_SET_SIZE_BLOCKS": {requestSetSizeBlock, 0xab07},
		"NBD_DISCONNECT":      {requestDisconnect, 0xab08},
		"NBD_SET_TIMEOUT":     {requestSetTimeout, 0xab09},
		"NBD_SET_FLAGS":       {requestSetFlags, 0xab0a},
	} {
		if tc.got != tc.want {
			t.Fatalf("%s = %#x, want %#x", name, tc.got, tc.want)
		}
	}
}
