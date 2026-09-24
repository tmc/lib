package trainstat_test

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/trainstat"
)

func Example() {
	w := trainstat.NewWriter(os.Stdout)
	w.Config("mode", "lora")
	w.Config("seed", "1")
	w.Step(1, trainstat.Value{V: 1.25, Unit: "loss"}, trainstat.Value{V: 2.5, Unit: "sec/step"})
	w.Final("split=val/group=all/cal=temp", trainstat.Value{V: 0.8119, Unit: "accuracy"})
	w.Done(trainstat.Run{Steps: 1, Seqs: 64, Tokens: 4096, TrainSec: 2.5, Done: true})
	if err := w.Err(); err != nil {
		fmt.Println(err)
	}
	// Output:
	// mode: lora
	// seed: 1
	// Unit loss better=lower assume=normal
	// Unit sec/step better=lower kind=time
	// BenchmarkTrain/step=1 1 1.25 loss 2.5 sec/step
	// Unit accuracy better=higher assume=normal delta=abs
	// BenchmarkFinal/split=val/group=all/cal=temp 1 0.8119 accuracy
	// Unit train-sec better=lower kind=time
	// Unit done better=higher delta=abs
	// Unit nonfinite better=lower delta=abs
	// BenchmarkRun 1 1 steps 64 seqs 4096 tokens 2.5 train-sec 1 done 0 nonfinite
}

func ExampleWriter_Step() {
	w := trainstat.NewWriter(os.Stdout)
	w.Unit("loss")
	for step, loss := range []float64{1.5, 0.75, math.NaN()} {
		w.Config("epoch", fmt.Sprint(1+step/2))
		w.Step(step+1, trainstat.Value{V: loss, Unit: "loss"})
	}
	// Output:
	// epoch: 1
	// BenchmarkTrain/step=1 1 1.5 loss
	// BenchmarkTrain/step=2 1 0.75 loss
	// epoch: 2
	// BenchmarkTrain/step=3 1 NaN loss
}

func ExampleWriter_Config() {
	w := trainstat.NewWriter(os.Stdout)
	w.Config("seed", "1")
	w.Result("Throughput", 100, trainstat.Value{V: 6809, Unit: "tokens/s"})
	w.Config("seed", "")
	w.Config("Seed", "2")
	fmt.Println(w.Err())
	// Output:
	// seed: 1
	// Unit tokens/s better=higher kind=time
	// BenchmarkThroughput 100 6809 tokens/s
	// trainstat: invalid configuration key "Seed"
}

func ExampleBuild() {
	for _, kv := range trainstat.Build() {
		if kv[0] == "go" {
			fmt.Println(kv[0])
		}
	}
	// Output: go
}

// attrs is a testing.TB that prints attributes.
type attrs struct{ testing.TB }

func (attrs) Helper()                {}
func (attrs) Attr(key, value string) { fmt.Println(key, value) }

func ExampleReport() {
	var t testing.TB = attrs{} // in a test, the *testing.T
	trainstat.Report(t, "", trainstat.Value{V: 0.4203, Unit: "loss"})
	trainstat.Report(t, "Final/split=val/group=all/cal=temp", trainstat.Value{V: 0.8119, Unit: "accuracy"})
	// Output:
	// trainstat:loss 0.4203
	// trainstat:Final/split=val/group=all/cal=temp:accuracy 0.8119
}
