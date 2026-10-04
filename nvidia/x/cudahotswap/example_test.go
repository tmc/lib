package cudahotswap_test

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

func Example() {
	c, err := cudahotswap.Compare([]float32{1, 2}, []float32{1, 2}, cudahotswap.Tolerance{Abs: 1e-6})
	fmt.Println(c.OK(), err)
	// Output: true <nil>
}

func ExampleCompare() {
	c, err := cudahotswap.Compare([]float32{2}, []float32{1}, cudahotswap.Tolerance{})
	fmt.Println(c.Elements, c.Mismatches, err)
	// Output: 1 1 <nil>
}

func ExampleComparison_OK() {
	c, _ := cudahotswap.Compare([]float32{1}, []float32{1}, cudahotswap.Tolerance{})
	fmt.Println(c.OK())
	// Output: true
}

func ExampleMismatch_MarshalJSON() {
	m := cudahotswap.Mismatch{Got: 1, Want: 0, AbsErr: 1, RelErr: math.Inf(1)}
	b, err := m.MarshalJSON()
	fmt.Println(string(b), err)
	// Output: {"index":0,"got":1,"want":0,"abs_err":1,"rel_err":"+Inf"} <nil>
}

func ExampleMismatch_UnmarshalJSON() {
	var m cudahotswap.Mismatch
	err := json.Unmarshal([]byte(`{"index":0,"got":"NaN","want":0,"abs_err":"NaN","rel_err":"+Inf"}`), &m)
	fmt.Println(math.IsNaN(m.Got), math.IsInf(m.RelErr, 1), err)
	// Output: true true <nil>
}

func ExampleClassify() {
	outcome, recognized := cudahotswap.Classify(driver.ErrIllegalAddress)
	fmt.Println(outcome, recognized)
	// Output: device_fault true
}

func ExampleCheckABI() {
	a := cudalaunch.Signature{Params: []cudalaunch.Param{{Type: "u64", Size: 8}}}
	b := cudalaunch.Signature{Params: []cudalaunch.Param{{Type: "u32", Size: 4}}}
	err := cudahotswap.CheckABI(a, b)
	outcome, recognized := cudahotswap.Classify(err)
	fmt.Println(outcome, recognized)
	// Output: abi_mismatch true
}

func ExampleOutcome_PoisonsContext() {
	fmt.Println(cudahotswap.OutcomeDeviceFault.PoisonsContext())
	// Output: true
}

func ExampleOutcome_ExitCode() {
	fmt.Println(cudahotswap.OutcomeABIMismatch.ExitCode())
	// Output: 1
}

func ExampleReport_Fail() {
	r := new(cudahotswap.Report).Fail(driver.ErrIllegalAddress)
	fmt.Println(r.Outcome, r.ContextPoisoned, r.Recognized)
	// Output: device_fault true true
}

func ExampleReport_Text() {
	r := cudahotswap.Report{Task: "copy", TaskDef: "copy one element", Candidate: "copy.cu", Outcome: cudahotswap.OutcomeOK, Recognized: true, Unmeasured: []string{"timing"}}
	fmt.Print(r.Text())
	// Output:
	// task      copy  (copy one element)
	// candidate copy.cu
	// outcome   ok
	// unmeasured:
	//   - timing
}

func ExampleTimingReport_Summarize() {
	r := cudahotswap.TimingReport{LaunchesPerSample: 4, SampleNanos: []int64{800, 1200, 1000}}
	r.Summarize()
	fmt.Println(r.Resolved, r.MedianNanosPerLaunch)
	// Output: true 250
}
