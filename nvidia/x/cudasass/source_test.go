package cudasass

import "testing"

func TestParseSourceDirective(t *testing.T) {
	tests := []struct {
		name string
		line string
		want SourceDirective
		ok   bool
	}{{
		name: "plain",
		line: `        //## File "kernels.cu", line 42`,
		want: SourceDirective{File: "kernels.cu", Line: 42, Kind: CUDASource},
		ok:   true,
	}, {
		name: "absolute path",
		line: `//## File "/home/user/cuda_moe_d2r.cu", line 7`,
		want: SourceDirective{File: "/home/user/cuda_moe_d2r.cu", Line: 7, Kind: CUDASource},
		ok:   true,
	}, {
		name: "inlined",
		line: `        //## File "util.cuh", line 7 inlined at "kernels.cu", line 42`,
		want: SourceDirective{File: "util.cuh", Line: 7, Kind: CUDASource, InlinedFile: "kernels.cu", InlinedLine: 42},
		ok:   true,
	}, {
		// The line number here is past the end of any real source; that is the
		// tell, and the reason this file must not share a namespace with one.
		name: "ptx text",
		line: `        //## File ".nv_debug_ptx_txt", line 146931`,
		want: SourceDirective{File: PTXTextFile, Line: 146931, Kind: PTXText},
		ok:   true,
	}, {
		// A real source whose name merely contains the pseudo-file's is a
		// source. A prefix or substring test would classify it as PTX text and
		// silently drop every instruction it accounts for.
		name: "source named like the pseudo file",
		line: `//## File ".nv_debug_ptx_txt.cu", line 3`,
		want: SourceDirective{File: ".nv_debug_ptx_txt.cu", Line: 3, Kind: CUDASource},
		ok:   true,
	}, {
		name: "instruction line",
		line: `        /*0000*/                   MOV R1, c[0x0][0x28] ;`,
	}, {
		name: "directive not at line start",
		line: `        MOV R1, R2 ; // see //## File "kernels.cu", line 42`,
	}, {
		name: "empty",
		line: "",
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ParseSourceDirective(tt.line)
			if ok != tt.ok {
				t.Fatalf("ok = %v, want %v", ok, tt.ok)
			}
			if got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}

// TestSourceDirectiveDiscriminates is the control for the classification: if
// Kind returned the same value for both kinds of file, every test above that
// checks a Kind would still pass on one of them alone.
func TestSourceDirectiveDiscriminates(t *testing.T) {
	src, _ := ParseSourceDirective(`//## File "a.cu", line 1`)
	ptx, _ := ParseSourceDirective(`//## File ".nv_debug_ptx_txt", line 1`)
	if src.Kind == ptx.Kind {
		t.Fatalf("both classified %v; the kind does not discriminate", src.Kind)
	}
	if src.Kind.String() == ptx.Kind.String() {
		t.Errorf("both render as %q", src.Kind)
	}
}

// TestPTXTextNotASourceFile pins the parser behaviour cudacapture relies on:
// the pseudo-file updates PTXLine and never lands in SourceFile, so a consumer
// joining on (file, line) cannot pick it up even if it forgets to exclude it.
func TestPTXTextNotASourceFile(t *testing.T) {
	const sass = `.section	.text.k,"ax",@progbits
        //## File "a.cu", line 12
        //## File ".nv_debug_ptx_txt", line 146931
        /*0000*/                   MOV R1, c[0x0][0x28] ;
`
	d, err := ParseString(sass)
	if err != nil {
		t.Fatal(err)
	}
	inst := d.Functions[0].Instructions[0]
	if inst.SourceFile != "a.cu" || inst.SourceLine != 12 {
		t.Errorf("source location = %s:%d, want a.cu:12", inst.SourceFile, inst.SourceLine)
	}
	if inst.PTXLine != 146931 {
		t.Errorf("PTXLine = %d, want 146931", inst.PTXLine)
	}
}
