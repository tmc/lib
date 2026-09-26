package cublaslog_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/cudaprof/cublaslog"
)

func Example() {
	m, err := cublaslog.Parse("A=0X0 Adesc=[type=R_16BF rows=1024 cols=768 ld=1024] B=0X0 Bdesc=[type=R_16BF rows=1024 cols=512 ld=1024] " +
		"C=0X0 Cdesc=[type=R_16BF rows=768 cols=512 ld=768] D=0X0 Ddesc=[type=R_16BF rows=768 cols=512 ld=768] " +
		"computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T epilogue=EPILOGUE_BIAS] " +
		"algo=[algoId=21 tile=MATMUL_TILE_16x16 stages=MATMUL_STAGES_32x1] beta=0 stream=0X0")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%dx%dx%d op=%s%s %s %s beta=%s algo %s\n", m.M, m.N, m.K, m.TransA, m.TransB, m.Types(), m.Epilogue, m.Beta, m.Algo)
	// Output:
	// 768x512x1024 op=TN R_16BF,R_16BF->R_16BF BIAS beta=0 algo 21 16x16/32x1
}
