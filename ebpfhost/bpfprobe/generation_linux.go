//go:build linux && (amd64 || arm64)

package bpfprobe

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"github.com/tmc/lib/ebpfhost/bpfload"
	"os"
)

type taskLayout struct {
	start, exec, execSize uint32
	hash                  string
}

func taskGenerationLayout() (taskLayout, error) {
	data, err := os.ReadFile("/sys/kernel/btf/vmlinux")
	if err != nil {
		return taskLayout{}, fmt.Errorf("bpf probe: kernel btf required: %w", err)
	}
	b, err := bpfload.ParseBTF(data)
	if err != nil {
		return taskLayout{}, err
	}
	matches := b.Find("task_struct", 4)
	if len(matches) != 1 {
		return taskLayout{}, errors.New("bpf probe: ambiguous or missing task_struct btf")
	}
	field := func(name string) (offset, size uint32, err error) {
		var fields []bpfload.BTFMember
		for _, m := range matches[0].Members {
			if m.Name == name {
				fields = append(fields, m)
			}
		}
		if len(fields) != 1 || fields[0].BitfieldSize != 0 || fields[0].BitOffset%8 != 0 {
			return 0, 0, fmt.Errorf("bpf probe: missing aligned task %s", name)
		}
		id := fields[0].Type
		for n := 0; n < 64; n++ {
			t, ok := b.Type(id)
			if !ok {
				break
			}
			if t.Kind == 1 {
				size = t.SizeOrType
				break
			}
			switch t.Kind {
			case 8, 9, 10, 11, 18:
				id = t.SizeOrType
			default:
				n = 64
			}
		}
		offset = fields[0].BitOffset / 8
		if size != 4 && size != 8 || uint64(offset)+uint64(size) > uint64(matches[0].SizeOrType) {
			return 0, 0, fmt.Errorf("bpf probe: task %s is not a bounded integer", name)
		}
		return offset, size, nil
	}
	start, startSize, err := field("start_boottime")
	if err != nil {
		return taskLayout{}, err
	}
	if startSize != 8 {
		return taskLayout{}, errors.New("bpf probe: task start_boottime is not a u64")
	}
	exec, size, err := field("self_exec_id")
	if err != nil {
		return taskLayout{}, err
	}
	return taskLayout{start, exec, size, fmt.Sprintf("%x", sha256.Sum256(data))}, nil
}
