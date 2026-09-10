.PHONY: all build test check

all: build test

build:
	@for dir in $$(go list -m -f '{{.Dir}}'); do \
		echo "==> Building $$dir"; \
		(cd "$$dir" && go build ./...) || exit 1; \
	done

test:
	@for dir in $$(go list -m -f '{{.Dir}}'); do \
		echo "==> Testing $$dir"; \
		(cd "$$dir" && go test ./...) || exit 1; \
	done

check:
	@missing=0; \
	for f in $$(git ls-files '*/go.mod'); do \
		d=$$(dirname "$$f"); \
		grep -qx "	./$$d" go.work || { echo "error: $$d is not listed in go.work"; missing=1; }; \
	done; \
	[ $$missing -eq 0 ] || exit 1; \
	echo "go.work covers every module."
	@for f in $$(git ls-files '*.go'); do \
		dir=$$(dirname "$$f"); \
		found=0; \
		while [ "$$dir" != "." ] && [ "$$dir" != "/" ]; do \
			if [ -f "$$dir/go.mod" ]; then \
				found=1; \
				break; \
			fi; \
			dir=$$(dirname "$$dir"); \
		done; \
		if [ $$found -eq 0 ]; then \
			echo "error: $$f is not part of any module"; \
			exit 1; \
		fi; \
	done; \
	echo "All Go files belong to a module."
