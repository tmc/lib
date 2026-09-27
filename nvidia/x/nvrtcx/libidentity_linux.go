package nvrtcx

import "os"

func mappedLibrary(soname string) string {
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return ""
	}
	return mappedLibraryIn(string(maps), soname)
}
