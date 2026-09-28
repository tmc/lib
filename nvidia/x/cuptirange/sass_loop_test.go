package cuptirange_test

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudasass"
)

func loopBodySASSInstructions(t *testing.T, dev *cuda.Device, ptx string) (int, error) {
	t.Helper()
	findTool := func(name string) (string, error) {
		for _, root := range []string{os.Getenv("CUDA"), os.Getenv("CUDA_HOME")} {
			if root == "" {
				continue
			}
			candidate := filepath.Join(root, "bin", name)
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
		}
		return exec.LookPath(name)
	}
	ptxas, err := findTool("ptxas")
	if err != nil {
		return 0, fmt.Errorf("find ptxas for static loop bound: %w", err)
	}
	nvdisasm, err := findTool("nvdisasm")
	if err != nil {
		return 0, fmt.Errorf("find nvdisasm for static loop bound: %w", err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return 0, err
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return 0, err
	}
	tmp := filepath.Join(home, "tmp")
	if err := os.MkdirAll(tmp, 0o700); err != nil {
		return 0, err
	}
	dir, err := os.MkdirTemp(tmp, "cuptirange-sass-")
	if err != nil {
		return 0, err
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	ptxPath := filepath.Join(dir, "range.ptx")
	cubinPath := filepath.Join(dir, "range.cubin")
	if err := os.WriteFile(ptxPath, []byte(ptx), 0o600); err != nil {
		return 0, err
	}
	compile := exec.Command(ptxas, "-arch=sm_"+fmt.Sprintf("%d%d", major, minor), "-lineinfo", "-o", cubinPath, ptxPath)
	if out, err := compile.CombinedOutput(); err != nil {
		return 0, fmt.Errorf("ptxas: %w: %s", err, out)
	}
	disassemble := exec.Command(nvdisasm, "-g", cubinPath)
	out, err := disassemble.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("nvdisasm: %w: %s", err, out)
	}
	parsed, err := cudasass.Parse(bytes.NewReader(out))
	if err != nil {
		return 0, fmt.Errorf("parse loop SASS: %w", err)
	}
	fn := parsed.Function("heavy_kernel")
	if fn == nil {
		return 0, fmt.Errorf("SASS omitted heavy_kernel")
	}
	return countBackedgeBlockInstructions(fn)
}

func countBackedgeBlockInstructions(fn *cudasass.Function) (int, error) {
	graph, err := cudasass.NewCFG(fn)
	if err != nil {
		return 0, fmt.Errorf("build SASS control-flow graph: %w", err)
	}
	if len(graph.Blocks) == 0 {
		return 0, fmt.Errorf("SASS function has no basic blocks")
	}
	reachable := make([]bool, len(graph.Blocks))
	reachable[0] = true
	work := []int{0}
	for len(work) != 0 {
		block := work[0]
		work = work[1:]
		for _, successor := range graph.Successors(block) {
			if !reachable[successor] {
				reachable[successor] = true
				work = append(work, successor)
			}
		}
	}
	count := 0
	for _, edge := range graph.Edges {
		if !reachable[edge.From] || edge.Kind != cudasass.Taken || edge.To > edge.From {
			continue
		}
		block := graph.Blocks[edge.From]
		if len(block.Instructions) == 0 || block.Instructions[len(block.Instructions)-1].Opcode == "" {
			return 0, fmt.Errorf("backedge block has no terminating instruction")
		}
		if count != 0 {
			return 0, fmt.Errorf("multiple SASS backedge blocks; static work is ambiguous")
		}
		count = len(block.Instructions)
	}
	if count == 0 {
		return 0, fmt.Errorf("SASS has no identifiable loop backedge block")
	}
	return count, nil
}

func TestCountBackedgeBlockInstructions(t *testing.T) {
	tests := []struct {
		name string
		sass string
		want int
	}{
		{
			name: "loop control on for line",
			sass: `.section .text.heavy_kernel,"ax",@progbits
.text.heavy_kernel:
/*0000*/ IADD R0, R0, 0x1 ;
.L_loop:
/*0010*/ ISETP.LT.AND P0, PT, R0, 0x10, PT ;
/*0020*/ IMAD R1, R1, 0x3, R0 ;
/*0030*/ @P0 BRA ` + "`" + `(.L_loop) ;
/*0040*/ EXIT ;
.L_trap:
/*0050*/ BRA ` + "`" + `(.L_trap) ;
/*0060*/ NOP ;
`,
			want: 3,
		},
		{
			name: "hoisted constant outside body",
			sass: `.section .text.heavy_kernel,"ax",@progbits
.text.heavy_kernel:
/*0000*/ MOV R2, 0x10 ;
.L_loop:
/*0010*/ ISETP.LT.AND P0, PT, R0, R2, PT ;
/*0020*/ IADD R0, R0, 0x1 ;
/*0030*/ @P0 BRA ` + "`" + `(.L_loop) ;
/*0040*/ EXIT ;
.L_trap:
/*0050*/ BRA ` + "`" + `(.L_trap) ;
/*0060*/ NOP ;
`,
			want: 3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dis, err := cudasass.ParseString(tt.sass)
			if err != nil {
				t.Fatal(err)
			}
			got, err := countBackedgeBlockInstructions(dis.Function("heavy_kernel"))
			if err != nil || got != tt.want {
				t.Fatalf("countBackedgeBlockInstructions = %d, %v; want %d", got, err, tt.want)
			}
		})
	}
}

func TestCountBackedgeBlockInstructionsFailsClosed(t *testing.T) {
	dis, err := cudasass.ParseString(`.section .text.heavy_kernel,"ax",@progbits
.text.heavy_kernel:
/*0000*/ EXIT ;
.L_trap:
/*0010*/ BRA ` + "`" + `(.L_trap) ;
/*0020*/ NOP ;
`)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := countBackedgeBlockInstructions(dis.Function("heavy_kernel")); err == nil {
		t.Fatal("SASS without a loop backedge was accepted")
	}
}

func TestCountBackedgeBlockInstructionsRejectsMultipleReachableLoops(t *testing.T) {
	dis, err := cudasass.ParseString(`.section .text.heavy_kernel,"ax",@progbits
.text.heavy_kernel:
/*0000*/ IADD R0, R0, 0x1 ;
.L_loop_a:
/*0010*/ ISETP.LT.AND P0, PT, R0, 0x10, PT ;
/*0020*/ @P0 BRA ` + "`" + `(.L_loop_a) ;
.L_loop_b:
/*0030*/ ISETP.LT.AND P1, PT, R1, 0x20, PT ;
/*0040*/ @P1 BRA ` + "`" + `(.L_loop_b) ;
/*0050*/ EXIT ;
.L_trap:
/*0060*/ BRA ` + "`" + `(.L_trap) ;
/*0070*/ NOP ;
`)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := countBackedgeBlockInstructions(dis.Function("heavy_kernel")); err == nil || !strings.Contains(err.Error(), "multiple SASS backedge blocks") {
		t.Fatalf("multiple reachable loops error = %v, want ambiguous-loop error", err)
	}
}

func TestCountBackedgeBlockInstructionsMatchesNvdisasmFixtures(t *testing.T) {
	tests := []struct {
		path, function, wantError string
	}{
		{"../cudasass/testdata/sm121a-encoding.sass", "addk", "no identifiable loop backedge"},
		{"../cudasass/testdata/sm121a-loop-13.4.sass", "k4", "multiple SASS backedge blocks"},
	}
	for _, tt := range tests {
		t.Run(tt.function, func(t *testing.T) {
			data, err := os.ReadFile(tt.path)
			if err != nil {
				t.Fatal(err)
			}
			dis, err := cudasass.Parse(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := countBackedgeBlockInstructions(dis.Function(tt.function)); err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("countBackedgeBlockInstructions error = %v, want %q", err, tt.wantError)
			}
		})
	}
}

func TestCountBackedgeBlockInstructionsHeavyKernelWithTrapBlock(t *testing.T) {
	dis, err := cudasass.ParseString(`.section .text.heavy_kernel,"ax",@progbits
.text.heavy_kernel:
/*0000*/ LDC R1, c[0x0][0x37c] ;
/*0010*/ S2R R0, SR_TID.X ;
/*0020*/ MOV R2, RZ ;
.L_x_0:
/*0030*/ ISETP.LT.AND P0, PT, R0, R1, PT ;
/*0040*/ IMAD R2, R2, 0x19660d, 0x3c6ef35f ;
/*0050*/ IADD R0, R0, 0x1 ;
/*0060*/ @P0 BRA ` + "`" + `(.L_x_0) ;
/*0070*/ STG.E [R4], R2 ;
/*0080*/ EXIT ;
.L_x_6:
/*0090*/ BRA ` + "`" + `(.L_x_6) ;
/*00a0*/ NOP ;
/*00b0*/ NOP ;
`)
	if err != nil {
		t.Fatal(err)
	}
	got, err := countBackedgeBlockInstructions(dis.Function("heavy_kernel"))
	if err != nil || got != 4 {
		t.Fatalf("heavy_kernel loop block size = %d, %v; want 4", got, err)
	}
}
