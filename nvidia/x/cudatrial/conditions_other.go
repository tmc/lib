//go:build !linux

package cudatrial

// observeHost reads nothing on hosts without /proc/meminfo: darwin reports no
// MemAvailable analogue, and a total-memory-only readout would invite the
// MemFree mistake it exists to avoid. Every host kind is left unread.
func observeHost(c *Conditions) {}
