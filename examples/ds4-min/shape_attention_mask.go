package main

import "errors"

func shapeAttentionUseMask(indexer bool, rows uint32) bool {
	return indexer && rows > 0
}

func appendShapeAttentionMaskRange(ranges []mxfp4DeviceRange, useMask bool, rows uint32, mask mxfp4DeviceRange) ([]mxfp4DeviceRange, error) {
	if !useMask {
		return ranges, nil
	}
	if rows == 0 || mask.bytes != uint64(rows)*4 {
		return nil, errors.New("cuda shape attention: invalid active mask range")
	}
	return append(ranges, mask), nil
}
