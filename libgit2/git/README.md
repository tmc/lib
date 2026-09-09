# Generated Go API

Import `github.com/tmc/lib/libgit2/git` for the generated owning API. The parent
package remains the low-level API.

```go
repo, err := git.Open(path)
if err != nil {
    return err
}
defer repo.Close()

id, err := repo.CreateBlob(data)
if err != nil {
    return err
}
blob, err := repo.LookupBlob(id)
if err != nil {
    return err
}
defer blob.Close()
contents, err := blob.Contents()
```

`Open` and `Init` acquire a libgit2 reference; repository `Close` releases it.
Close blobs before their repository. Repository `Close` returns `ErrChildren`
when a blob remains open, and calls after close return `ErrClosed`. Repeated
close is harmless. `Contents` copies native bytes into Go memory.

Errors include native Git status, class, and a copied message. The failing call
and message extraction run on one locked OS thread. Use `errors.As` with
`*git.Error`. Loading errors retain their original error through wrapping.

Resources and their children must not be used concurrently. Do not call the raw
package's `Close` or manually shut down references owned by this package.

Generation is opt-in:

```sh
go -C ../../../codegen run ./cmd/gen-libgit2 \
  -header "$HOME/tmp/libgit2/include/git2.h" -output ../lib/libgit2 \
  -facade-import github.com/tmc/lib/libgit2
```

The semantic policy and resource template live in `contrib/libgit2/facade.go`
and `facade.go.tmpl`. Generation rejects missing or changed C declarations.
No generated file should be edited directly.

This first surface covers repositories, blobs, SHA-1 OIDs, and errors. Commits,
references, clone/fetch, option conversion, and Go callback bridges remain to
be modeled. It is not a complete git2go-compatible API.
