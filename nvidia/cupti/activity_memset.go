package cupti

// Layout size constants for CUPTI ActivityMemset record versions, measured with
// the compiler against the toolkit headers on an sbsa-linux host:
//
//	gcc -I/usr/local/cuda-13.X/targets/sbsa-linux/include, sizeof(CUpti_ActivityMemset{4,5})
const (
	ActivityMemset4Size = 88  // CUDA 13.0–13.3
	ActivityMemset5Size = 104 // CUDA 13.4
)
