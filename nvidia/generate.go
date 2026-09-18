package nvidia

// The packages beneath this directory are generated. Regenerating all of them
// is one command:
//
//	go generate
//
// That runs nvgen once per package, each invocation reading the materialized
// CUDA profile named by $CUDA_PROFILE and writing one directory. nvgen holds
// the recipe -- which generator runs with which arguments -- so this file
// lists only which packages exist and which of them the CUDA profile selects.
//
// nvgen lives in the generator repository, which this module does not depend
// on. Install it from a sibling checkout before generating:
//
//	(cd ../../libgen && go install ./cmd/nvgen)
//
// Choosing a profile. The profile is a materialized header tree under
// cuda-headers-profiles beside this checkout, selected by name:
//
//	CUDA_PROFILE=cuda-12.6.3 go generate
//
// It defaults to the latest pinned patch profile. The packages in the first
// block below name that tree in their generator arguments, so a toolkit bump
// moves them and only them. The second block names no profile path at all:
// those inputs ship with their own NVIDIA SDK, which no CUDA version selects.
//
// Checking without writing. To report what a regeneration would change
// without touching the tree:
//
//	nvgen -diff -output .
//
// A clean -diff means every file nvgen owns matches its inputs. This is the
// check to run in CI; it is not run by go generate. nvgen -list prints all 35
// recipes, marking one whose gate is unset as skipped rather than omitting it.
//
// What nvgen does not cover. Two packages are generated entirely elsewhere:
// x/cudamlx and x/cudaptx carry their own go:generate directives, because
// their inputs are not the CUDA profile. See their package docs.
//
// Three more packages are generated from here but hold one file that is not.
// nvjitlink, nvrtc and nvvm each take flags.gen.go from genoptions, whose
// input is optiondata rather than the CUDA profile. nvgen neither writes nor
// diffs those three files, so -diff reports a clean package while one of them
// has drifted; regenerate with go generate in the package itself, or run
// TestFlagsGenUpToDate, which regenerates all three and compares.
//
// Today x is part of this module, so that regeneration needs nothing special.
// When x becomes its own module it will need the workspace: the command is then
// in another module, go.work is what lets this one reach it, and it must stay
// that way, because a require on x would put x in every consumer's graph.

//go:generate -command nvgen nvgen -output .

// Generated from the CUDA profile: a toolkit bump moves these.
//go:generate nvgen driver
//go:generate nvgen nvml
//go:generate nvgen nvrtc
//go:generate nvgen cublas
//go:generate nvgen cublaslt
//go:generate nvgen cufft
//go:generate nvgen curand
//go:generate nvgen cusolver
//go:generate nvgen cusparse
//go:generate nvgen cufile
//go:generate nvgen cudnn
//go:generate nvgen cutensor
//go:generate nvgen fabricmanager
//go:generate nvgen nccl
//go:generate nvgen nvfatbin
//go:generate nvgen nvjitlink
//go:generate nvgen nvvm
//go:generate nvgen sanitizer
//go:generate nvgen cublasmp
//go:generate nvgen cudensitymat
//go:generate nvgen cudss
//go:generate nvgen cusolvermp
//go:generate nvgen custatevec
//go:generate nvgen cutensornet
//go:generate nvgen nvshmem
//go:generate nvgen cupti

// Independent NVIDIA SDK inputs; the CUDA profile does not select these.
//go:generate nvgen dcgm
//go:generate nvgen gdrcopy
//go:generate nvgen ncclep
//go:generate nvgen ncclm2n
//go:generate nvgen nvcomp
//go:generate nvgen nvdec
//go:generate nvgen nvenc
//go:generate nvgen nvjpeg2k

// VPI headers ship with JetPack rather than the CUDA toolkit, so this recipe
// runs only where $VPI_INCLUDE names a materialized tree. nvgen skips it
// otherwise; naming it explicitly reports why instead.
//go:generate nvgen vpi
