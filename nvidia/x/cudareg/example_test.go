package cudareg_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudareg"
)

func ExampleGenerateFixture() {
	s, src, _ := cudareg.GenerateFixture()
	fmt.Println(s.Target.Name, len(s.Steps), len(src) > 0) /* Output: sm_90a 2 true */
}
func ExampleTargetFor() {
	t, _ := cudareg.TargetFor("sm_100a")
	fmt.Println(t.Name, len(t.LegalCounts)) /* Output: sm_100a 30 */
}
func ExampleSchedule_Validate() {
	s, _, _ := cudareg.GenerateFixture()
	fmt.Println(s.Validate()) /* Output: <nil> */
}
func ExampleReference()         { fmt.Println(cudareg.Reference([]uint32{4, 5})) /* Output: [5 6] */ }
func ExampleFixedBudgetSource() { fmt.Println(len(cudareg.FixedBudgetSource()) > 0) /* Output: true */ }
