// One Llama MLP sub-block (hidden 256, intermediate 512, 4 tokens), as
// jax.jit(...).lower(...).compiler_ir("stablehlo") prints it, with the
// weights stored [in, out].
module @jit_mlp {
  func.func public @main(%arg0: tensor<4x256xbf16>, %arg1: tensor<256xbf16>, %arg2: tensor<256x512xbf16>, %arg3: tensor<256x512xbf16>, %arg4: tensor<512x256xbf16>) -> (tensor<4x256xf32> {jax.result_info = ""}) {
    %0 = stablehlo.convert %arg0 : (tensor<4x256xbf16>) -> tensor<4x256xf32>
    %1 = stablehlo.multiply %0, %0 : tensor<4x256xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x256xf32>, tensor<f32>) -> tensor<4xf32>
    %cst_0 = stablehlo.constant dense<2.560000e+02> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %4 = stablehlo.divide %2, %3 : tensor<4xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %5 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %6 = stablehlo.add %4, %5 : tensor<4xf32>
    %7 = stablehlo.rsqrt %6 : tensor<4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<4xf32>) -> tensor<4x256xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<4x256xf32>
    %10 = stablehlo.convert %arg1 : (tensor<256xbf16>) -> tensor<256xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<256xf32>) -> tensor<4x256xf32>
    %12 = stablehlo.multiply %9, %11 : tensor<4x256xf32>
    %13 = stablehlo.convert %12 : (tensor<4x256xf32>) -> tensor<4x256xbf16>
    %14 = stablehlo.dot_general %13, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x256xbf16>, tensor<256x512xbf16>) -> tensor<4x512xf32>
    %15 = stablehlo.logistic %14 : tensor<4x512xf32>
    %16 = stablehlo.multiply %14, %15 : tensor<4x512xf32>
    %17 = stablehlo.dot_general %13, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x256xbf16>, tensor<256x512xbf16>) -> tensor<4x512xf32>
    %18 = stablehlo.multiply %16, %17 : tensor<4x512xf32>
    %19 = stablehlo.convert %18 : (tensor<4x512xf32>) -> tensor<4x512xbf16>
    %20 = stablehlo.dot_general %19, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x512xbf16>, tensor<512x256xbf16>) -> tensor<4x256xf32>
    %21 = stablehlo.add %20, %0 : tensor<4x256xf32>
    return %21 : tensor<4x256xf32>
  }
}
