// Command modelir-run inspects ModelIR descriptions and executes ModelIR graphs.
//
// It supports shebang-executable model definitions, S-expression files (.sexpr),
// and native ModelIR text files (.modelir). Graph execution uses the pure-Go
// ModelIR reference interpreter and explicit JSON tensors.
//
// # Usage
//
//	modelir-run [flags] <model.sexpr|model.modelir>
//	modelir-run -inputs inputs.json model.modelir
//
// # Flags
//
//	-function <name>   function to execute (default: model entry)
//	-inputs <path>     JSON object of input tensors
//	-weights <path>    JSON object of constant tensors
//
// # Tensor files
//
// Each tensor file is a JSON object. Every tensor has a shape and row-major
// data, for example: {"x":{"Shape":[2],"Data":[1.5,3]}}.
package main
