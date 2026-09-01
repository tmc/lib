package metalref

// BlockReduceSum computes the scalar sum over a float32 slice.
func BlockReduceSum(data []float32) float32 {
	var sum float32
	for _, v := range data {
		sum += v
	}
	return sum
}
