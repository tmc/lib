package main

// batchQ81FFNNeeded reports whether the batched Q8_1 FFN activation is an
// input to a remaining prefill route. The D2R paths consume q81MMQ instead;
// when both routed MoE and shared dense D2R are active, q81FFN has no reader.
// Keep the fallback cases explicit: either route being disabled must retain
// the quantization for the ordinary expert or shared projection path.
func batchQ81FFNNeeded(gather, moeD2R, denseD2R bool, denseGateBytes int) bool {
	return !(gather && moeD2R && denseD2R && denseGateBytes > 0)
}
