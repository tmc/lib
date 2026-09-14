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
// A clean -diff means every committed package matches its inputs. This is the
// check to run in CI; it is not run by go generate. Note that nvgen -list
// prints only the recipes whose gates are satisfied, so it reports 34 of the
// 35 directives below on a host without JetPack.
//
// Two packages are not generated from here. x/cudamlx and x/cudaptx carry
// their own go:generate directives, because their inputs are not the CUDA
// profile. nvgen neither generates nor checks them; see their package docs.

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
