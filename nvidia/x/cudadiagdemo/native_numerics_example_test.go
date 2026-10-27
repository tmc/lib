package cudadiagdemo

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/tmc/lib/nvidia/x/cudaregress"
)

func ExampleNativeNumerics() {
	// Host fakes stand in for the device streams; the executor, encoding
	// and export are RunNative's with an OutputDir. This is not native
	// execution.
	dir, err := os.MkdirTemp("", "cudadiagdemo-numerics-")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(dir)
	e := (&fakeNative{}).executor()
	e.retain = true
	attempts, err := e.run(context.Background(), 1, planNativeRequests(1, 2), &nativeCleanup{})
	if err != nil {
		fmt.Println(err)
		return
	}
	result := &NativeResult{}
	result.recordAttempts(attempts)
	if err := exportNative(dir, result, nil); err != nil {
		fmt.Println(err)
		return
	}

	// Verify the retained bytes from the receipt alone.
	data, err := os.ReadFile(filepath.Join(dir, "native-receipt.json"))
	if err != nil {
		fmt.Println(err)
		return
	}
	var receipt struct {
		Numerics NativeNumerics `json:"numerics"`
	}
	if err := json.Unmarshal(data, &receipt); err != nil {
		fmt.Println(err)
		return
	}
	decode := func(ref *cudaregress.ArtifactRef) []float32 {
		b, err := os.ReadFile(ref.Path)
		if err != nil || fmt.Sprintf("%x", sha256.Sum256(b)) != ref.Digest || int64(len(b)) != ref.SizeBytes {
			return nil
		}
		values := make([]float32, len(b)/4)
		if err := binary.Read(bytes.NewReader(b), binary.LittleEndian, values); err != nil {
			return nil
		}
		return values
	}
	fmt.Println(receipt.Numerics.Retention)
	for _, v := range receipt.Numerics.Vectors {
		in, ref, out := decode(v.Input), decode(v.Reference), decode(v.Output)
		ok := len(in) > 0 && len(ref) == len(in) && len(out) == len(in)
		for i, x := range in {
			ok = ok && ref[i] == 2*x+1 && out[i] == ref[i]
		}
		if !ok {
			fmt.Printf("%s/%s: unverified\n", v.RequestID, v.Attempt)
			continue
		}
		fmt.Printf("%s/%s: %v -> %v verified\n", v.RequestID, v.Attempt, in[:2], out[:2])
	}
	// Output:
	// retained
	// request-1/baseline: [0 1] -> [1 3] verified
	// request-1/candidate: [1 2] -> [3 5] verified
	// request-1/device: [2 3] -> [5 7] verified
}
