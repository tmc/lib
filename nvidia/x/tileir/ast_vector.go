package tileir

// vectorWidth admits the bounded power-of-two rank-one tiles used by this
// route. The bound is a source construction limit, not a GPU resource receipt.
func vectorWidth(shape []int64) bool {
	return len(shape) == 1 && shape[0] > 0 && shape[0] <= 1024 && shape[0]&(shape[0]-1) == 0
}

func checkVectorOperands(name string, result Type, operands []Type) (bool, error) {
	switch name {
	case "get_tile_block_id":
		tuple, ok := result.(*FunctionType)
		if !ok || tuple == nil || len(tuple.Results) != 3 || len(operands) != 0 {
			return true, astUnsupported("block ID requires three results")
		}
		for _, typ := range tuple.Results {
			if !scalarMemoryType(typ, KindInt32) {
				return true, astUnsupported("block ID results must be scalar tile<i32>")
			}
		}
		return true, nil
	case "iota":
		tile, ok := result.(*TileType)
		if !ok || tile == nil || !vectorWidth(tile.Shape) || tile.ElementType.Kind() != KindInt64 || len(operands) != 0 {
			return true, astUnsupported("vector iota requires rank-one tile<i64>")
		}
		return true, nil
	case "reshape", "broadcast", "exti", "offset":
	default:
		return false, nil
	}
	to, ok := result.(*TileType)
	if !ok || to == nil || len(operands) == 0 {
		return true, astUnsupported("vector operation needs tile types")
	}
	from, ok := operands[0].(*TileType)
	if !ok || from == nil {
		return true, astUnsupported("vector source type")
	}
	switch name {
	case "reshape":
		a, err := astElements(from)
		if err != nil {
			return true, err
		}
		b, err := astElements(to)
		if err != nil {
			return true, err
		}
		if len(operands) != 1 || a != b || !typeEqual(from.ElementType, to.ElementType) {
			return true, astUnsupported("reshape changes elements or element type")
		}
	case "broadcast":
		if len(operands) != 1 || len(from.Shape) != len(to.Shape) || !typeEqual(from.ElementType, to.ElementType) {
			return true, astUnsupported("broadcast changes rank or element type")
		}
		for i, dim := range from.Shape {
			if dim != to.Shape[i] && dim != 1 {
				return true, astUnsupported("broadcast incompatible dimensions")
			}
		}
	case "exti":
		if len(operands) != 1 || !sameASTShape(from, to) || from.ElementType.Kind() != KindInt32 || to.ElementType.Kind() != KindInt64 {
			return true, astUnsupported("vector extension must be i32 to i64 with equal shape")
		}
	case "offset":
		if len(operands) != 2 || !typeEqual(from, to) {
			return true, astUnsupported("pointer offset source/result mismatch")
		}
		if err := checkMemoryPointer(to); err != nil {
			return true, err
		}
		offset, ok := operands[1].(*TileType)
		if !ok || offset == nil || offset.ElementType.Kind() != KindInt64 || !sameASTShape(offset, to) {
			return true, astUnsupported("pointer offset needs same-shape i64 indices")
		}
	}
	return true, nil
}
