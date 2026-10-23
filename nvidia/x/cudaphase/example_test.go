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
