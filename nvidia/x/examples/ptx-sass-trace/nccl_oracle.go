package main

import (
	"fmt"
)

// ncclInput returns the exact int32 input value for rank r and element i:
// input[r][i] = 1024*(r+1) + (i mod 1024)
func ncclInput(r, i int) int32 {
	return int32(1024*(r+1) + (i % 1024))
}

// allReduceExpected computes the expected AllReduce sum across R ranks:
// 1024*R*(R+1)/2 + R*(i mod 1024)
func allReduceExpected(ranks, i int) (int32, error) {
	R := int64(ranks)
	sum := 1024*R*(R+1)/2 + R*int64(i%1024)
	if sum < -1<<31 || sum > 1<<31-1 {
		return 0, fmt.Errorf("nccl allreduce: sum %d overflows int32", sum)
	}
	return int32(sum), nil
}

// allGatherExpected returns the expected AllGather result for index idx in [0, R*N):
// at offset r*N + i, original input is of rank r.
func allGatherExpected(elements, idx int) int32 {
	r := idx / elements
	i := idx % elements
	return ncclInput(r, i)
}
