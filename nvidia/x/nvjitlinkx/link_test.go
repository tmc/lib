package nvjitlinkx

import (
	"errors"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/nvjitlink"
)

type fakeLink struct {
	fail   string
	err    error
	closed int
	data   []byte
	image  []byte
}

func (f *fakeLink) at(stage string) error {
	if f.fail == stage {
		return f.err
	}
	return nil
}
func (f *fakeLink) AddData(_ nvjitlink.InputType, p unsafe.Pointer, n uint, _ string) error {
	f.data = unsafe.Slice((*byte)(p), int(n))
	return f.at("add")
}
func (f *fakeLink) Complete() error                 { return f.at("complete") }
func (f *fakeLink) GetLinkedCubin() ([]byte, error) { return f.image, f.at("output") }
func (f *fakeLink) GetLinkedPtx() ([]byte, error)   { return f.image, f.at("output") }
func (f *fakeLink) GetErrorLog() (string, error)    { return "diagnostic", f.at("log") }
func (f *fakeLink) GetInfoLog() (string, error)     { return "info", nil }
func (f *fakeLink) close() error {
	f.closed++
	if err := f.at("close"); err != nil {
		return err
	}
	for i := range f.image {
		f.image[i] = 0
	}
	f.data = nil
	return nil
}

func TestLinkLifecycle(t *testing.T) {
	for _, stage := range []string{"", "create", "add", "complete", "output", "log", "close"} {
		t.Run(stage, func(t *testing.T) {
			want := errors.New("injected failure")
			f := &fakeLink{fail: stage, err: want, image: []byte{1, 2, 3}}
			input := []byte("ptx")
			result, err := link([]Input{{Name: "input", Kind: nvjitlink.InputPtx, Data: input}}, []string{"-arch=sm_80"}, func([]string) (linkHandle, error) {
				if stage == "create" {
					return nil, want
				}
				input[0] = 'X' // Inputs must already be copied.
				return f, nil
			})
			if stage != "" && !errors.Is(err, want) {
				t.Fatalf("got %v, want injected error", err)
			}
			if stage == "" {
				if err != nil {
					t.Fatal(err)
				}
				if len(result.CUBIN) != 3 || result.CUBIN[0] != 1 {
					t.Fatal("output did not survive native cleanup")
				}
			}
			if stage == "create" {
				if f.closed != 0 {
					t.Fatal("closed uncreated handle")
				}
				return
			}
			if result.ErrorLog != "diagnostic" || result.InfoLog != "info" {
				t.Fatal("lost diagnostics")
			}
			if f.closed != 1 {
				t.Fatalf("close calls = %d", f.closed)
			}
			if stage == "close" {
				if string(f.data) != "ptx\x00" {
					t.Fatal("input copy or PTX terminator missing")
				}
				f.fail = ""
			}
			if err := result.Close(); err != nil {
				t.Fatal(err)
			}
			count := f.closed
			if err := result.Close(); err != nil || f.closed != count {
				t.Fatal("Close is not idempotent")
			}
		})
	}
}

func TestLinkValidation(t *testing.T) {
	valid := Input{Name: "x", Kind: nvjitlink.InputPtx, Data: []byte("ptx")}
	for _, tt := range []struct {
		name    string
		inputs  []Input
		options []string
	}{
		{"empty", nil, []string{"-arch=sm_80"}},
		{"target", []Input{valid}, nil},
		{"ptx without lto", []Input{valid}, []string{"-arch=sm_80", "-ptx"}},
		{"unknown kind", []Input{{Name: "x", Kind: 99, Data: []byte{1}}}, []string{"-arch=sm_80"}},
		{"lto without option", []Input{{Name: "x", Kind: nvjitlink.InputLtoir, Data: []byte{1}}}, []string{"-arch=sm_80"}},
		{"nul option", []Input{valid}, []string{"-arch=sm_80\x00bad"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := link(tt.inputs, tt.options, func([]string) (linkHandle, error) { t.Fatal("invalid input reached driver"); return nil, nil })
			if err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
}
