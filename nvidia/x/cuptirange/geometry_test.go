package cuptirange

import (
	"testing"
	"unsafe"
)

func TestDirectLaunchGeometry(t *testing.T) {
	p := launchKernelParams{GridX: 4, GridY: 2, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1, SharedMemory: 64}
	grid, block, shared, ok := directLaunchGeometry(unsafe.Pointer(&p))
	if !ok || grid != [3]uint32{4, 2, 1} || block != [3]uint32{128, 1, 1} || shared != 64 {
		t.Fatalf("geometry = %v, %v, %d, %v", grid, block, shared, ok)
	}
	if unsafe.Sizeof(p) != 64 || unsafe.Offsetof(p.Stream) != 40 || unsafe.Offsetof(p.Params) != 48 {
		t.Fatal("unqualified launch parameter layout")
	}
	p.BlockX = 0
	if _, _, _, ok := directLaunchGeometry(unsafe.Pointer(&p)); ok {
		t.Fatal("accepted zero geometry")
	}
	if _, _, _, ok := directLaunchGeometry(nil); ok {
		t.Fatal("invented absent geometry")
	}
}

func TestKernelReplayGeometryWitness(t *testing.T) {
	k := &kernelNames{context: 7, limit: 1}
	p := launchKernelParams{GridX: 2, GridY: 1, GridZ: 1, BlockX: 128, BlockY: 1, BlockZ: 1}
	evt := launchEvent("gemv")
	evt.Api.FunctionParams = unsafe.Pointer(&p)
	if err := k.begin(); err != nil {
		t.Fatal(err)
	}
	k.callback(evt)
	if err := k.end(); err != nil {
		t.Fatal(err)
	}
	ranges := []RangeInfo{{Name: "0"}}
	if err := k.resolve(ranges); err != nil {
		t.Fatal(err)
	}
	if !ranges[0].GeometryAvailable || ranges[0].Grid != [3]uint32{2, 1, 1} || ranges[0].Block != [3]uint32{128, 1, 1} {
		t.Fatalf("native range=%+v", ranges)
	}
	p.GridX++
	if err := k.begin(); err != nil {
		t.Fatal(err)
	}
	k.callback(evt)
	if err := k.end(); err == nil {
		t.Fatal("accepted changed native geometry across replay")
	}
}
