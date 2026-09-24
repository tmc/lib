package main

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"encoding/json"
	"strconv"
	"testing"

	"github.com/google/pprof/profile"
)

// microKernelSample builds one short kernel activity sample carrying device
// timestamps and a launch correlation ID. extraValues pads the sample to a
// profile with more than one sample type.
func microKernelSample(name string, stream int, start, end uint64, correlation int, extraValues ...int64) *profile.Sample {
	value := append([]int64{int64(end - start)}, extraValues...)
	return &profile.Sample{
		Value: value,
		Label: map[string][]string{
			"name":    {name},
			"subtype": {"kernel"},
			"stream":  {strconv.Itoa(stream)},
		},
		NumLabel: map[string][]int64{
			"start":       {int64(start)},
			"end":         {int64(end)},
			"correlation": {int64(correlation)},
		},
	}
}

// apiRecordsFile renders host launch API records, keyed by correlation ID, into
// the capture-status.json form advise reads them from.
func apiRecordsFile(t *testing.T, starts map[int]uint64) []byte {
	t.Helper()
	records := make([]map[string]any, 0, len(starts))
	for corr, start := range starts {
		records = append(records, map[string]any{"domain": "driver", "correlation": corr, "start": start, "end": start + 500})
	}
	data, err := json.Marshal(map[string]any{"api_records": records})
	if err != nil {
		t.Fatalf("marshal api records: %v", err)
	}
	return data
}

func synthesizeTestELFForAdvise(sections map[string][]byte) []byte {
	var buf bytes.Buffer

	secNames := []string{""}
	secData := [][]byte{nil}
	secTypes := []uint32{0}
	secFlags := []uint64{0}

	shstrtabIdx := len(secNames)
	secNames = append(secNames, ".shstrtab")
	secTypes = append(secTypes, uint32(elf.SHT_STRTAB))
	secFlags = append(secFlags, 0)
	secData = append(secData, nil)

	for name, data := range sections {
		secNames = append(secNames, name)
		secTypes = append(secTypes, uint32(elf.SHT_PROGBITS))
		secFlags = append(secFlags, 0)
		secData = append(secData, data)
	}

	var shstrtab bytes.Buffer
	shstrtab.WriteByte(0)
	nameOffsets := make([]uint32, len(secNames))
	for i, name := range secNames {
		if i == 0 {
			nameOffsets[i] = 0
			continue
		}
		nameOffsets[i] = uint32(shstrtab.Len())
		shstrtab.WriteString(name)
		shstrtab.WriteByte(0)
	}
	secData[shstrtabIdx] = shstrtab.Bytes()

	ehdr := make([]byte, 64)
	copy(ehdr[0:4], []byte("\x7fELF"))
	ehdr[4] = 2                                     // 64-bit
	ehdr[5] = 1                                     // Little-endian
	ehdr[6] = 1                                     // Version 1
	binary.LittleEndian.PutUint16(ehdr[16:18], 1)   // ET_REL
	binary.LittleEndian.PutUint16(ehdr[18:20], 190) // EM_CUDA
	binary.LittleEndian.PutUint32(ehdr[20:24], 1)   // EV_CURRENT
	binary.LittleEndian.PutUint16(ehdr[52:54], 64)  // ehsize
	binary.LittleEndian.PutUint16(ehdr[58:60], 64)  // shentsize
	binary.LittleEndian.PutUint16(ehdr[60:62], uint16(len(secNames)))
	binary.LittleEndian.PutUint16(ehdr[62:64], uint16(shstrtabIdx))

	buf.Write(ehdr)

	secOffsets := make([]uint64, len(secNames))
	for i, data := range secData {
		if len(data) == 0 {
			secOffsets[i] = 0
			continue
		}
		secOffsets[i] = uint64(buf.Len())
		buf.Write(data)
	}

	shoff := uint64(buf.Len())
	binary.LittleEndian.PutUint64(ehdr[40:48], shoff)
	copy(buf.Bytes()[40:48], ehdr[40:48])

	for i := range secNames {
		shdr := make([]byte, 64)
		binary.LittleEndian.PutUint32(shdr[0:4], nameOffsets[i])
		binary.LittleEndian.PutUint32(shdr[4:8], secTypes[i])
		binary.LittleEndian.PutUint64(shdr[8:16], secFlags[i])
		binary.LittleEndian.PutUint64(shdr[24:32], secOffsets[i])
		binary.LittleEndian.PutUint64(shdr[32:40], uint64(len(secData[i])))
		binary.LittleEndian.PutUint64(shdr[48:56], 1)
		buf.Write(shdr)
	}

	return buf.Bytes()
}
