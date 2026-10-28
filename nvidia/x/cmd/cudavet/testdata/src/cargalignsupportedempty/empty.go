package cargalignsupportedempty

import "github.com/tmc/lib/nvidia/x/cudakernel"

var a, _ = cudakernel.NewSchema()
var b, _ = cudakernel.NewSchema([]cudakernel.Param{}...)
var c, _ = cudakernel.NewSchema(nil...)
var d, _ = cudakernel.NewSchema(([]cudakernel.Param{})...)
var e = cudakernel.Schema{}
var f = cudakernel.Schema{Params: []cudakernel.Param{}, TotalParamBytes: 0}
var g = cudakernel.Schema{Params: nil}
