// Package nvfatbinx packages alternative CUDA device images into a fatbinary.
// Build copies its inputs and returns owned output. Packaging does not resolve
// device symbols or fuse kernels; use nvjitlinkx for linking. The loaded library
// validates architecture and format compatibility. TileIR requires a library
// exporting nvFatbinAddTileIR. No CUDA device context is needed for packaging.
package nvfatbinx
