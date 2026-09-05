package nvfatbinx

import (
	"bytes"
	"errors"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/nvfatbin"
)

var errInjected = errors.New("injected")

type fakeHandle struct {
	fail   string
	inputs []Input
	data   []byte
	closes int
}

func (h *fakeHandle) add(input Input) error {
	h.inputs = append(h.inputs, input)
	if h.fail == "add" {
		return errInjected
	}
	if h.fail == "symbol" {
		return &nvfatbin.SymbolError{Symbol: "nvFatbinAddTileIR"}
	}
	return nil
}
func (h *fakeHandle) Get() ([]byte, error) {
	if h.fail == "get" {
		return nil, errInjected
	}
	return h.data, nil
}
func (h *fakeHandle) close() error {
	h.closes++
	if h.fail == "close" {
		return errInjected
	}
	clear(h.data)
	return nil
}

func TestBuildLifecycle(t *testing.T) {
	for _, fail := range []string{"", "create", "add", "get", "empty", "close", "symbol"} {
		t.Run(fail, func(t *testing.T) {
			h := &fakeHandle{fail: fail, data: []byte{1, 0, 2}}
			if fail == "empty" {
				h.data = nil
			}
			input := Input{Kind: TileIR, Name: "tile", Data: []byte{3, 0, 4}}
			r, err := build([]Input{input}, nil, func([]string) (handle, error) {
				if fail == "create" {
					return nil, errInjected
				}
				return h, nil
			})
			if (err != nil) != (fail != "") {
				t.Fatalf("error = %v", err)
			}
			if fail == "create" {
				if r != nil {
					t.Fatal("result on creation failure")
				}
				return
			}
			if h.closes != 1 {
				t.Fatalf("closes = %d", h.closes)
			}
			if fail == "symbol" {
				var symbol *nvfatbin.SymbolError
				if !errors.As(err, &symbol) || symbol.Symbol != "nvFatbinAddTileIR" {
					t.Fatalf("lost symbol error: %v", err)
				}
			}
			if fail == "close" {
				if r.resources.pins == nil || r.resources.h == nil {
					t.Fatal("lost resources after cleanup error")
				}
				h.fail = ""
				var wg sync.WaitGroup
				for range 8 {
					wg.Go(func() {
						if err := r.Close(); err != nil {
							t.Error(err)
						}
					})
				}
				wg.Wait()
				if h.closes != 2 {
					t.Fatalf("closes = %d", h.closes)
				}
			}
			if r.resources.h != nil || r.resources.pins != nil {
				t.Fatal("resources retained after destruction")
			}
			if fail == "" {
				if !bytes.Equal(r.Data, []byte{1, 0, 2}) {
					t.Fatal("output invalidated by destroy")
				}
				input.Data[0] = 9
				if h.inputs[0].Data[0] != 3 {
					t.Fatal("input was not copied")
				}
			}
		})
	}
}

func TestBuildFormats(t *testing.T) {
	inputs := []Input{
		{Kind: CUBIN, Name: "cubin", Arch: "80", Data: []byte{0, 1}},
		{Kind: PTX, Name: "ptx", Arch: "90", Data: []byte("ptx")},
		{Kind: LTOIR, Name: "lto", Arch: "100a", Data: []byte{0, 2}},
		{Kind: TileIR, Name: "tile", Data: []byte{0, 3}},
	}
	h := &fakeHandle{data: []byte{1}}
	if _, err := build(inputs, nil, func([]string) (handle, error) { return h, nil }); err != nil {
		t.Fatal(err)
	}
	if len(h.inputs) != 4 || string(h.inputs[1].Data) != "ptx\x00" {
		t.Fatal("lost format or PTX terminator")
	}
	for i := range inputs {
		if h.inputs[i].Kind != inputs[i].Kind || h.inputs[i].Arch != inputs[i].Arch {
			t.Fatal("format metadata changed")
		}
	}
}

func TestBuildValidation(t *testing.T) {
	valid := Input{Kind: PTX, Name: "ptx", Arch: "80", Data: []byte("ptx")}
	for _, tt := range []struct {
		name    string
		change  func(*Input)
		options []string
	}{
		{"kind", func(i *Input) { i.Kind = 0 }, nil},
		{"name", func(i *Input) { i.Name = "" }, nil},
		{"data", func(i *Input) { i.Data = nil }, nil},
		{"arch", func(i *Input) { i.Arch = "" }, nil},
		{"tile arch", func(i *Input) { i.Kind = TileIR }, nil},
		{"cubin options", func(i *Input) { i.Kind = CUBIN; i.JITOptions = "-g" }, nil},
		{"nul", func(i *Input) { i.Name = "x\x00" }, nil},
		{"option", func(*Input) {}, []string{"-g\x00"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			input := valid
			tt.change(&input)
			if _, err := build([]Input{input}, tt.options, func([]string) (handle, error) { t.Fatal("native entry before validation"); return nil, nil }); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}
