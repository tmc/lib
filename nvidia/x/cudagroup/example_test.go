package cudagroup_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagroup"
)

func Example() {
	g, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	rank, _ := g.Rank(driver.Dim3{X: 7})
	fmt.Println(rank, g.Size())
	// Output: 7 32
}

func ExampleNew() {
	g, err := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	fmt.Println(err, g.Kind(), g.Dims(), g.Size()) /* Output: <nil> warp 32x1x1 32 */
}
func ExampleGroup_Rank() {
	g, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	rank, err := g.Rank(driver.Dim3{X: 7})
	fmt.Println(rank, err, g.Contains(driver.Dim3{X: 31})) /* Output: 7 <nil> true */
}
func ExampleGroup_Size() {
	g, _ := cudagroup.New(cudagroup.CTA, driver.Dim3{X: 128, Y: 2, Z: 1})
	fmt.Println(g.Size()) /* Output: 256 */
}
func ExampleGroup_Dims() {
	g, _ := cudagroup.New(cudagroup.Thread, driver.Dim3{X: 1, Y: 1, Z: 1})
	fmt.Println(g.Dims()) /* Output: 1x1x1 */
}
func ExampleGroup_Kind() {
	g, _ := cudagroup.New(cudagroup.WarpGroup, driver.Dim3{X: 128, Y: 1, Z: 1})
	fmt.Println(g.Kind()) /* Output: warpgroup */
}
func ExampleWarpGroupForCTA() {
	cta, _ := cudagroup.New(cudagroup.CTA, driver.Dim3{X: 256, Y: 1, Z: 1})
	g, _ := cudagroup.WarpGroupForCTA(cta, 4)
	rank, _ := g.Rank(driver.Dim3{})
	fmt.Println(g.Kind(), g.Size(), rank) /* Output: warpgroup 128 128 */
}

func ExampleNewCollective() {
	warp, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	c, _ := cudagroup.NewCollective(cudagroup.ReduceSum, warp, cudagroup.MemoryThread)
	ranks := make([]uint64, 32)
	for i := range ranks {
		ranks[i] = uint64(i)
	}
	fmt.Println(c.ValidateRanks(ranks), c.ValidateParticipation(32)) /* Output: <nil> <nil> */
}
func ExampleCollective_ValidateRanks() {
	warp, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	c, _ := cudagroup.NewCollective(cudagroup.Broadcast, warp, cudagroup.MemoryCTA)
	ranks := make([]uint64, 32)
	for i := range ranks {
		ranks[i] = uint64(i)
	}
	fmt.Println(c.ValidateRanks(ranks)) /* Output: <nil> */
}
func ExampleCollective_ValidateUniform() {
	warp, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	c, _ := cudagroup.NewCollective(cudagroup.ReduceSum, warp, cudagroup.MemoryThread)
	ranks, values := make([]uint64, 32), make([]uint32, 32)
	for i := range ranks {
		ranks[i] = uint64(i)
		values[i] = 7
	}
	fmt.Println(c.ValidateUniform(ranks, values)) /* Output: <nil> */
}

func ExampleNewTMEMCollective() {
	cluster, _ := cudagroup.New(cudagroup.Cluster, driver.Dim3{X: 2, Y: 1, Z: 1})
	t, _ := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup2, cluster, driver.Dim3{X: 128, Y: 1, Z: 1}, 0, 0)
	fmt.Println(t.Peers()) /* Output: 2 */
}
func ExampleTMEMCollective_Peers() {
	cluster, _ := cudagroup.New(cudagroup.Cluster, driver.Dim3{X: 1, Y: 1, Z: 1})
	t, _ := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup1, cluster, driver.Dim3{X: 128, Y: 1, Z: 1}, 0, 0)
	fmt.Println(t.Peers() == cudagroup.TMEMGroup1) /* Output: true */
}
func ExampleTMEMCollective_ValidateLaunch() {
	cluster, _ := cudagroup.New(cudagroup.Cluster, driver.Dim3{X: 1, Y: 1, Z: 1})
	t, _ := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup1, cluster, driver.Dim3{X: 128, Y: 1, Z: 1}, 0, 0)
	err := t.ValidateLaunch(cudagroup.LaunchRequirements{Grid: driver.Dim3{X: 1, Y: 1, Z: 1}, Block: driver.Dim3{X: 128, Y: 1, Z: 1}, Cluster: driver.Dim3{X: 1, Y: 1, Z: 1}})
	fmt.Println(err) /* Output: <nil> */
}
func ExampleTMEMCollective_ValidateParticipation() {
	cluster, _ := cudagroup.New(cudagroup.Cluster, driver.Dim3{X: 1, Y: 1, Z: 1})
	t, _ := cudagroup.NewTMEMCollective("sm_100a", cudagroup.TMEMGroup1, cluster, driver.Dim3{X: 32, Y: 1, Z: 1}, 0, 0)
	issuers, args := make([]cudagroup.Issuer, 32), make([]uint32, 32)
	for i := range issuers {
		issuers[i] = cudagroup.Issuer{CTA: 0, Thread: uint64(i)}
		args[i] = 8
	}
	fmt.Println(t.ValidateParticipation(issuers, args)) /* Output: <nil> */
}

func ExampleFixture_ValidateLaunch() {
	warp, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	f, _ := cudagroup.GenerateReductionFixture(warp)
	fmt.Println(f.Target, f.ValidateLaunch()) /* Output: sm_90 <nil> */
}
func ExampleReduceSumReference() {
	values := make([]float32, 32)
	for i := range values {
		values[i] = 1
	}
	sum, _ := cudagroup.ReduceSumReference(values)
	fmt.Println(sum) /* Output: 32 */
}
func ExampleGenerateReductionFixture() {
	warp, _ := cudagroup.New(cudagroup.Warp, driver.Dim3{X: 32, Y: 1, Z: 1})
	f, _ := cudagroup.GenerateReductionFixture(warp)
	fmt.Println(f.Collective.Group.Size(), len(f.Source) > 0) /* Output: 32 true */
}
