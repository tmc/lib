//go:build !linux

package cupticallback

func currentThreadID() uint64 {
	return 1
}
