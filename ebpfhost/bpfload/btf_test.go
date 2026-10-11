package bpfload

import (
	"encoding/binary"
	"os"
	"runtime"
	"testing"
)

func tinyBTF(order binary.ByteOrder) []byte {
	words := []uint32{1, 1 << 24, 4, 32, 5, 4<<24 | 1, 4, 11, 1, 0}
	strings := []byte("\x00int\x00point\x00x\x00")
	out := make([]byte, 24+len(words)*4+len(strings))
	order.PutUint16(out, 0xeb9f)
	out[2] = 1
	order.PutUint32(out[4:], 24)
	order.PutUint32(out[12:], uint32(len(words)*4))
	order.PutUint32(out[16:], uint32(len(words)*4))
	order.PutUint32(out[20:], uint32(len(strings)))
	for i, w := range words {
		order.PutUint32(out[24+i*4:], w)
	}
	copy(out[24+len(words)*4:], strings)
	return out
}
func TestBTF(t *testing.T) {
	for _, order := range []binary.ByteOrder{binary.LittleEndian, binary.BigEndian} {
		data := tinyBTF(order)
		b, err := ParseBTF(data)
		if err != nil {
			t.Fatal(err)
		}
		found := b.Find("point", 4)
		if len(found) != 1 || found[0].SizeOrType != 4 || found[0].Members[0].Name != "x" || found[0].Members[0].Type != 1 {
			t.Fatalf("layout: %+v", found)
		}
		found[0].Members[0].Name = "corrupted"
		actual, _ := b.Type(2)
		if actual.Members[0].Name != "x" {
			t.Fatal("type aliases mutable input")
		}
		for n := 0; n < len(data); n++ {
			if _, err := ParseBTF(data[:n]); err == nil {
				t.Fatalf("accepted prefix %d", n)
			}
		}
		broken := append([]byte(nil), data...)
		order.PutUint32(broken[24+16+12+4:], 99)
		if _, err := ParseBTF(broken); err == nil {
			t.Fatal("accepted invalid member type")
		}
	}
}
func TestNativeKernelBTF(t *testing.T) {
	if runtime.GOOS != "linux" || os.Getenv("EBPFHOST_NATIVE") != "1" {
		t.Skip("native BTF admission is opt-in")
	}
	data, err := os.ReadFile("/sys/kernel/btf/vmlinux")
	if err != nil {
		t.Fatal(err)
	}
	b, err := ParseBTF(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.Types()) < 100 {
		t.Fatal("unexpectedly small kernel BTF")
	}
	t.Logf("parsed %d kernel BTF records; function presence does not prove kfunc registration", len(b.Types()))
}
