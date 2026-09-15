package cuptibenchexample

import "math"

// reference uses deterministic float32 inputs and double-precision accumulation
// to check the GPU reductions independently. No model file is required.
func reference(width, rows int) (input, weights, norm, projection, output []float32) {
	input, weights = make([]float32, width), make([]float32, rows*width)
	norm, projection, output = make([]float32, width), make([]float32, rows), make([]float32, rows)
	var squares float64
	for i := range input {
		input[i] = float32(i%29-14) / 16
		squares += float64(input[i]) * float64(input[i])
	}
	inv := 1 / math.Sqrt(squares/float64(width)+1e-5)
	for i, x := range input {
		norm[i] = float32(float64(x) * inv)
	}
	for row := 0; row < rows; row++ {
		var sum float64
		for col, x := range norm {
			w := float32((row*17+col*13)%31-15) / 512
			weights[row*width+col] = w
			sum += float64(w) * float64(x)
		}
		projection[row] = float32(sum)
		x := float64(projection[row])
		output[row] = float32(x / (1 + math.Exp(-x)))
	}
	return
}
