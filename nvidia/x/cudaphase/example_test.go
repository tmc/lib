package cudaphase_test

import (
	"fmt"
	"log"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaphase"
)

func ExampleSummarize() {
	// Two blocks, 3 slots each: start, main loop done, end (ns).
	buf := []uint64{
		1000, 4000, 5000,
		1256, 5256, 5512,
	}
	s, err := cudaphase.Summarize("gemm", buf, 3, []cudaphase.Phase{{"main", 0, 1}, {"epilogue", 1, 2}}, -1)
	if err != nil {
		log.Fatal(err)
	}
	for _, p := range s.Phases {
		fmt.Printf("%s: median %.3f us, max %.3f us\n", p.Name, p.MedianUS, p.MaxUS)
	}
	fmt.Printf("span %.3f us, timer step %.3f us\n", s.SpanUS, s.QuantumUS)
	// Output:
	// main: median 3.000 us, max 4.000 us
	// epilogue: median 0.256 us, max 1.000 us
	// span 4.512 us, timer step 0.256 us
}

func ExampleSummarizeClock() {
	// Two blocks, 6 slots each: clock64 at start, after the main loop and
	// at end; globaltimer at start and end; SM. Both run at 1.5 GHz.
	buf := []uint64{
		9000, 12000, 13500, 1000000, 1003000, 1,
		500, 4100, 5000, 1000256, 1003256, 2,
	}
	c := cudaphase.Clock{Start: 0, End: 2, TimerStart: 3, TimerEnd: 4, SMEnd: -1}
	s, err := cudaphase.SummarizeClock("gemm", buf, 6, []cudaphase.Phase{{"main", 0, 1}, {"epilogue", 1, 2}}, 5, c)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%.2f GHz from %d blocks on %d SMs\n", s.GHz, s.RateBlocks, s.SMs)
	for _, p := range s.Phases {
		fmt.Printf("%s: median %.3f us, max %.3f us\n", p.Name, p.MedianUS, p.MaxUS)
	}
	fmt.Printf("span %.3f us\n", s.SpanUS)
	// Output:
	// 1.50 GHz from 2 blocks on 2 SMs
	// main: median 2.000 us, max 2.400 us
	// epilogue: median 0.600 us, max 1.000 us
	// span 3.256 us
}

func ExampleParse() {
	judgeLog := `ok 5/5
CUDAPHASE {"name":"attn","blocks":264,"span_us":21.5,"quantum_us":0.256,"sms":132,"phases":[{"name":"main","blocks":264,"median_us":9.2,"min_us":7.9,"max_us":12.0}]}
`
	sums, err := cudaphase.Parse(strings.NewReader(judgeLog))
	if err != nil {
		log.Fatal(err)
	}
	for _, s := range sums {
		fmt.Printf("%s: %d blocks on %d SMs, main median %.1f us\n", s.Name, s.Blocks, s.SMs, s.Phases[0].MedianUS)
	}
	// Output: attn: 264 blocks on 132 SMs, main median 9.2 us
}
