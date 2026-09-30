package cupti

// Layout size constants for CUPTI ActivityMemcpy record versions, measured with
// the compiler against the toolkit headers on an sbsa-linux host:
//
//	gcc -I/usr/local/cuda-13.X/targets/sbsa-linux/include, sizeof(CUpti_ActivityMemcpy{6,7})
const (
	ActivityMemcpy6Size = 96  // CUDA 13.0–13.3
	ActivityMemcpy7Size = 112 // CUDA 13.4
)
