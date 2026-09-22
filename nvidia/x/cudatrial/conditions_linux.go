package cudatrial

// meminfoPath is the file observeHost reads.
const meminfoPath = "/proc/meminfo"

func observeHost(c *Conditions) {
	_ = observeMeminfoFile(meminfoPath, c)
}
