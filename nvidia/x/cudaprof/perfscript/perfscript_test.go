package perfscript

import (
	"reflect"
	"strings"
	"testing"
)

const script = `cudaprof  803190 64745.948787865: 
	          59d470 compress/flate.(*decompressor).huffmanBlock (/bin/cudaprof)
	          4c86a8 io.ReadAll+0x48 (/bin/cudaprof)

sys1 train  12/13 64745.949000000:      1001 cpu-clock:u: 
	ffff0001 [unknown] ([unknown])
	ffff0002 cuStreamSynchronize (/usr/lib/libcuda.so.1)

x 7 1.5:
`

func TestParse(t *testing.T) {
	got, err := Parse(strings.NewReader(script))
	if err != nil {
		t.Fatal(err)
	}
	want := []Sample{
		{Time: 64745948787865, Thread: 803190, Comm: "cudaprof", Stack: []string{"compress/flate.(*decompressor).huffmanBlock", "io.ReadAll"}},
		{Time: 64745949000000, Thread: 13, Comm: "sys1 train", Stack: []string{"[unknown]", "cuStreamSynchronize"}},
		{Time: 1500000000, Thread: 7, Comm: "x"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Parse =\n%+v\nwant\n%+v", got, want)
	}
}

func TestParseErrors(t *testing.T) {
	tests := []struct {
		name, in, want string
	}{
		{"no time", "cudaprof 12\n", "no tid and time"},
		{"orphan frame", "\t abc f (x)\n", "frame outside a sample"},
		{"bad tid", "c x 1.0:\n", "bad tid"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Parse(strings.NewReader(tt.in))
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Errorf("err = %v, want %q", err, tt.want)
			}
		})
	}
}
