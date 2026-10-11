package cudatracehost

import (
	"debug/elf"
	"testing"
)

func TestImageABI(t *testing.T) {
	for _, tt := range []struct {
		arch    string
		machine elf.Machine
		class   elf.Class
		data    elf.Data
		good    bool
	}{
		{"amd64", elf.EM_X86_64, elf.ELFCLASS64, elf.ELFDATA2LSB, true},
		{"arm64", elf.EM_AARCH64, elf.ELFCLASS64, elf.ELFDATA2LSB, true},
		{"arm64", elf.EM_ARM, elf.ELFCLASS32, elf.ELFDATA2LSB, false},
		{"arm64", elf.EM_X86_64, elf.ELFCLASS64, elf.ELFDATA2LSB, false},
		{"arm64", elf.EM_AARCH64, elf.ELFCLASS64, elf.ELFDATA2MSB, false},
	} {
		if err := validateImageABI(tt.class, tt.data, tt.machine, tt.arch); (err == nil) != tt.good {
			t.Fatalf("%+v error%v", tt, err)
		}
	}
}
