package dsv4vision

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func ExampleAttentionBounds() {
	tokens := []int32{1, 100, 102, 104, 2}
	b, err := AttentionBounds(tokens, 100, 0, 5, 128)
	fmt.Println(b, err)
	// Output: [[0 0] [0 3] [0 3] [0 3] [0 4]] <nil>
}

func ExamplePrefillChunk() {
	n, err := PrefillChunk([]int32{1, 100, 102, 104, 2}, 100, 0, 3)
	fmt.Println(n, err)
	// Output: 1 <nil>
}

func TestAttentionPinnedC(t *testing.T) {
	src := os.Getenv("DS4_VISION_ORACLE_SOURCE")
	if src == "" {
		t.Skip("requires frozen C source")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("host test requires gpu-lock")
	}
	const rev = "11e8cad4473716a41baf4d62e92ed9764a00fce6"
	head, err := exec.Command("git", "-C", src, "rev-parse", "HEAD").Output()
	if err != nil || strings.TrimSpace(string(head)) != rev {
		t.Fatal("wrong oracle revision")
	}
	diff, err := exec.Command("git", "-C", src, "diff", "HEAD", "--", "ds4_image.c", "ds4_image.h", "third_party/iris").Output()
	if err != nil || len(diff) != 0 {
		t.Fatal("dirty oracle")
	}
	dir := t.TempDir()
	for name, want := range map[string]string{
		"ds4_image.c":             "3aeb188835b41cb572413dcaac52555d53c65f6994ae45c0efa92ed2b3aecd68",
		"ds4_image.h":             "3ecf0420f52f8cd0ab69f325cafb183be868ee046ce04f703147dd08de067007",
		"third_party/iris/jpeg.h": "7d1bf2f6ce88232f6e8bb2bebcfd787d6c2f28e8afcfb9e676590e7906d77719",
		"third_party/iris/png.h":  "b87dda8580af164c443e76ca073d77f1340e1a7584b73a8449a0f5bf14cd5018",
	} {
		b, err := exec.Command("git", "-C", src, "show", rev+":"+name).Output()
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(b)) != want {
			t.Fatalf("source hash %s", name)
		}
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	bin := filepath.Join(dir, "attention-oracle")
	if out, err := exec.Command("cc", "-O2", "-I", dir, "testdata/attention_oracle.c", filepath.Join(dir, "ds4_image.c"), "-lm", "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("C compile: %v %s", err, out)
	}
	cases := 0
	for _, hw := range [][2]int{{1, 1}, {2, 3}, {3, 2}, {9, 9}, {11, 29}} {
		for start := 0; start < 4; start++ {
			types, _, err := Layout(hw[0], hw[1], start)
			if err != nil {
				t.Fatal(err)
			}
			tokens := make([]int32, start)
			for _, typ := range types {
				tokens = append(tokens, 100+int32(typ))
			}
			tokens = append(tokens, 1, 2)
			var in bytes.Buffer
			_ = binary.Write(&in, binary.LittleEndian, uint32(len(tokens)))
			_ = binary.Write(&in, binary.LittleEndian, tokens)
			for _, window := range []int{0, 1, 128} {
				for _, raw := range []int{len(tokens), len(tokens) / 2} {
					cmd := exec.Command(bin, "bounds", "100", "256", strconv.Itoa(raw), strconv.Itoa(window))
					cmd.Stdin = bytes.NewReader(in.Bytes())
					out, err := cmd.Output()
					if err != nil {
						t.Fatal(err)
					}
					got, err := AttentionBounds(tokens, 100, 256, raw, window)
					if err != nil {
						t.Fatal(err)
					}
					var b bytes.Buffer
					_ = binary.Write(&b, binary.LittleEndian, got)
					if !bytes.Equal(out, b.Bytes()) {
						t.Fatalf("C bounds mismatch grid%v start%d raw%d window%d", hw, start, raw, window)
					}
					cases++
				}
			}
			for _, cap := range []int{1, 4, len(tokens)} {
				cmd := exec.Command(bin, "chunk", "100", "0", strconv.Itoa(cap), "0")
				cmd.Stdin = bytes.NewReader(in.Bytes())
				out, cerr := cmd.Output()
				got, gerr := PrefillChunk(tokens, 100, 0, cap)
				checkOracleExit(t, cerr)
				if (cerr != nil) != (gerr != nil) {
					t.Fatalf("C chunk rejection differs: grid%v start%d cap%d", hw, start, cap)
				}
				if cerr == nil && (len(out) != 4 || int(binary.LittleEndian.Uint32(out)) != got) {
					t.Fatal("C chunk count mismatch")
				}
				cases++
			}
		}
	}
	extra := [][]int32{
		{1, -1}, {102}, {101, 101}, {100, 102}, {100, 100, 104},
		{100, 105, 104}, {100, 2, 104},
		{1, 2, 3},
		{1, 100, 102, 104, 2, 101, 100, 102, 104, 3},
	}
	// The grammar accepts these stress spans even though Layout limits actual
	// model image blocks to 384 tokens. They exercise the literal 383/384 caps.
	large := []int32{1, 100}
	for i := 0; i < 900; i++ {
		large = append(large, 102)
	}
	large = append(large, 104, 2)
	extra = append(extra, large)
	for ci, tokens := range extra {
		var in bytes.Buffer
		_ = binary.Write(&in, binary.LittleEndian, uint32(len(tokens)))
		_ = binary.Write(&in, binary.LittleEndian, tokens)
		for _, window := range []int{0, 1, 128, 384} {
			cmd := exec.Command(bin, "bounds", "100", "256", strconv.Itoa(len(tokens)), strconv.Itoa(window))
			cmd.Stdin = bytes.NewReader(in.Bytes())
			out, cerr := cmd.Output()
			got, gerr := AttentionBounds(tokens, 100, 256, len(tokens), window)
			checkOracleExit(t, cerr)
			if (cerr != nil) != (gerr != nil) {
				t.Fatalf("C bounds rejection differs extra%d window%d", ci, window)
			}
			var b bytes.Buffer
			_ = binary.Write(&b, binary.LittleEndian, got)
			if cerr == nil && !bytes.Equal(out, b.Bytes()) {
				t.Fatalf("C bounds mismatch extra%d window%d", ci, window)
			}
			cases++
		}
		for _, cap := range []int{1, 4, len(tokens)} {
			cmd := exec.Command(bin, "chunk", "100", "0", strconv.Itoa(cap), "0")
			cmd.Stdin = bytes.NewReader(in.Bytes())
			out, cerr := cmd.Output()
			got, gerr := PrefillChunk(tokens, 100, 0, cap)
			checkOracleExit(t, cerr)
			if (cerr != nil) != (gerr != nil) {
				t.Fatalf("C chunk rejection differs extra%d cap%d", ci, cap)
			}
			if cerr == nil && (len(out) != 4 || int(binary.LittleEndian.Uint32(out)) != got) {
				t.Fatalf("C chunk count mismatch extra%d cap%d", ci, cap)
			}
			cases++
		}
	}
	t.Logf("actual pinned C bounds/chunk cases=%d", cases)
}

func checkOracleExit(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		return
	}
	if exit, ok := err.(*exec.ExitError); !ok || exit.ExitCode() != 3 {
		t.Fatalf("oracle invocation: %v", err)
	}
}

func TestOracleExitControl(t *testing.T) {
	if code := os.Getenv("DS4_VISION_EXIT_CONTROL"); code != "" {
		err := exec.Command("sh", "-c", "exit "+code).Run()
		checkOracleExit(t, err)
		return
	}
	checkOracleExit(t, nil)
	checkOracleExit(t, exec.Command("sh", "-c", "exit 3").Run())
	for _, code := range []string{"1", "7", "127"} {
		t.Run(code, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=^TestOracleExitControl$")
			cmd.Env = append(os.Environ(), "DS4_VISION_EXIT_CONTROL="+code)
			out, err := cmd.CombinedOutput()
			exit, ok := err.(*exec.ExitError)
			if !ok || exit.ExitCode() != 1 || !strings.Contains(string(out), "oracle invocation:") {
				t.Fatalf("wrong oracle exit accepted or unexpected failure: %v %s", err, out)
			}
		})
	}
}
