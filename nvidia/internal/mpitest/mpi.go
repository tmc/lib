//go:build linux && cgo && nvidiampi

// Package mpitest provides native MPI objects for opt-in distributed tests.
package mpitest

/*
#include <mpi.h>
#include <stdlib.h>
static int start(int *rank, int *size, MPI_Comm **comm, int *free_err, int *finalize_err) {
 int provided;
 int duplicated = 0;
 int err = MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
 *comm = NULL;
 *free_err = MPI_SUCCESS;
 *finalize_err = MPI_SUCCESS;
 if (err != MPI_SUCCESS) return err;
 if (provided < MPI_THREAD_FUNNELED) { err = MPI_ERR_OTHER; goto fail; }
 *comm = malloc(sizeof(MPI_Comm));
 if (!*comm) { err = MPI_ERR_OTHER; goto fail; }
 err = MPI_Comm_dup(MPI_COMM_WORLD, *comm);
 if (err != MPI_SUCCESS) goto fail;
 duplicated = 1;
 err = MPI_Comm_set_errhandler(**comm, MPI_ERRORS_RETURN);
 if (err != MPI_SUCCESS) goto fail;
 err = MPI_Comm_rank(**comm, rank);
 if (err != MPI_SUCCESS) goto fail;
 err = MPI_Comm_size(**comm, size);
 if (err == MPI_SUCCESS) return err;
fail:
 if (duplicated) *free_err = MPI_Comm_free(*comm);
 free(*comm);
 *comm = NULL;
 *finalize_err = MPI_Finalize();
 return err;
}
static int gather(MPI_Comm *comm, void *src, int count, void *dst) {
 return MPI_Allgather(src, count, MPI_BYTE, dst, count, MPI_BYTE, *comm);
}
static int barrier(MPI_Comm *comm) { return MPI_Barrier(*comm); }
static int finish(MPI_Comm *comm) {
 int err = MPI_Comm_free(comm);
 free(comm);
 int last = MPI_Finalize();
 return err ? err : last;
}
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

type Comm struct {
	ptr        *C.MPI_Comm
	Rank, Size int
}

func result(code C.int) error {
	if code != C.MPI_SUCCESS {
		return fmt.Errorf("MPI error %d", int(code))
	}
	return nil
}

// Init must run on a locked OS thread; all calls must use that thread.
func Init() (*Comm, error) {
	var rank, size, freeErr, finalizeErr C.int
	var ptr *C.MPI_Comm
	if err := result(C.start(&rank, &size, &ptr, &freeErr, &finalizeErr)); err != nil {
		return nil, errors.Join(err, result(freeErr), result(finalizeErr))
	}
	return &Comm{ptr: ptr, Rank: int(rank), Size: int(size)}, nil
}
func (c *Comm) Pointer() unsafe.Pointer { return unsafe.Pointer(c.ptr) }
func (c *Comm) Bytes() uint             { return uint(C.sizeof_MPI_Comm) }
func (c *Comm) Barrier() error          { return result(C.barrier(c.ptr)) }
func (c *Comm) AllGather(src, dst []byte) error {
	if len(src) == 0 || len(dst) != len(src)*c.Size {
		return fmt.Errorf("invalid gather sizes")
	}
	return result(C.gather(c.ptr, unsafe.Pointer(&src[0]), C.int(len(src)), unsafe.Pointer(&dst[0])))
}
func (c *Comm) Close() error { return result(C.finish(c.ptr)) }
