# Regeneration and API-compatibility targets for the nvidia module.
#
# version-check is deliberately not named release-check: the root Makefile
# owns release-check, and that target is a license and module-metadata gate
# that says nothing about the Go API.

REPO := $(abspath ..)
TAG ?= $(shell git -C $(REPO) tag -l 'nvidia/v*' | sort -V | tail -1)
WORK ?= $(HOME)/tmp/nvidia-apidiff
LOG := $(REPO)/api/nvidia.log

# apidiff writes one "Ignoring internal package" line per internal package to
# stdout. Those are not API entries and are not kept in the committed report.
# They are filtered out of the comparison only; the -w steps print them, which
# is noisy but keeps apidiff's exit status from being hidden behind a pipe.
FILTER := grep -v '^Ignoring internal package '

.PHONY: regen regen-check version-check api-update api-added provenance-check clean

# Regenerate every binding package, then check what the regeneration did to
# the API. The check runs here rather than in a person's memory: a rename
# that leaves constant values alone is invisible to every other gate.
#
# The recipes live in generate.go, which every clone has. They replace
# regen.sh, which was excluded by .git/info/exclude and so was present in
# one working copy and in no clone.
regen:
	@command -v nvgen >/dev/null || { \
		echo "regen: nvgen is not installed." >&2; \
		echo "It lives in the generator repository, which this module does" >&2; \
		echo "not depend on: (cd ../../libgen && go install ./cmd/nvgen)" >&2; \
		exit 1; \
	}
	go generate .
	$(MAKE) version-check

# Report what a regeneration would change without writing to the tree. This
# is the check to run in CI; neither regen nor go generate runs it.
regen-check:
	nvgen -diff -output .

# Fail if the module's incompatible-change set against $(TAG) differs from
# the committed report. An entry is reviewed when it is in that report.
#
# Both sides are sorted before comparing. apidiff emits its entries in an
# order that is not stable across runs or platforms, so an unsorted diff
# reports a difference on every invocation. Sorting separates the warning's
# continuation lines from their entry, which makes the diff a set difference
# and not a readable report; read api/nvidia.log itself for that.
# A verdict has to name a revision, so a dirty tree fails: this working copy
# is shared, and a gate that reads uncommitted work reports on a state no tag
# will ever have. Set ALLOW_DIRTY=1 to check anyway; the verdict then says so.
version-check: $(WORK)/old.api $(WORK)/new.api
	@rev=$$(git -C $(REPO) rev-parse --short HEAD); \
	if [ -n "$$(git -C $(REPO) status --porcelain)" ]; then \
		if [ "$(ALLOW_DIRTY)" = 1 ]; then \
			echo "version-check: DIRTY TREE at $$rev; this verdict names no revision"; \
		else \
			echo "version-check: working tree is dirty at $$rev." >&2; \
			echo "Commit, or set ALLOW_DIRTY=1 to check the tree as it is." >&2; \
			exit 1; \
		fi; \
	else \
		echo "version-check: clean tree at $$rev"; \
	fi
	@GOWORK=off apidiff -m -incompatible $(WORK)/old.api $(WORK)/new.api \
		> $(WORK)/raw.log
	@$(FILTER) < $(WORK)/raw.log | sort > $(WORK)/current.log
	@sort $(LOG) > $(WORK)/recorded.log
	@if diff -u $(WORK)/recorded.log $(WORK)/current.log; then \
		echo "version-check: unchanged against $(TAG)"; \
	else \
		echo "version-check: API differs from api/nvidia.log; review the" >&2; \
		echo "diff above, then run 'make api-update' to record it." >&2; \
		exit 1; \
	fi

# Record the current set as reviewed. Commit the result with the change that
# caused it, never on its own.
# provenance-check verifies the generated provenance records. It is not part
# of the release gate yet: all 28 records predate libgen's bdf4b61 redaction
# and fail properties 1-3 today. It joins version-check in the commit that
# regenerates them, which is when it first passes.
provenance-check:
	GOWORK=off go run ./internal/cmd/provenancecheck

api-update: $(WORK)/old.api $(WORK)/new.api
	GOWORK=off apidiff -m -incompatible $(WORK)/old.api $(WORK)/new.api \
		> $(WORK)/raw.log
	$(FILTER) < $(WORK)/raw.log > $(LOG)

# Export data for the tag, taken from a detached worktree so the working
# copy is untouched. Delete it to re-take it.
$(WORK)/old.api:
	@mkdir -p $(WORK)
	rm -rf $(WORK)/tag
	git -C $(REPO) worktree add -q --detach $(WORK)/tag $(TAG)
	cd $(WORK)/tag/nvidia && GOWORK=off apidiff -m -w $@ github.com/tmc/lib/nvidia
	git -C $(REPO) worktree remove --force $(WORK)/tag

# Export data for the working tree. Always re-taken.
.PHONY: $(WORK)/new.api
$(WORK)/new.api:
	@mkdir -p $(WORK)
	GOWORK=off apidiff -m -w $@ github.com/tmc/lib/nvidia

# What the incompatible set cannot show. api/nvidia.log is -incompatible
# output, so every addition is missing from it -- and this release is mostly
# additions. Print the other half so a changelog written from the log alone
# does not silently omit the larger story.
api-added: $(WORK)/old.api $(WORK)/new.api
	@GOWORK=off apidiff -m $(WORK)/old.api $(WORK)/new.api > $(WORK)/full.log
	@echo "compatible entries: $$(awk '/^Compatible changes:/{f=1;next} f' \
		$(WORK)/full.log | grep -c '^- ')"
	@echo "packages added:"
	@awk '/^Compatible changes:/{f=1;next} f' $(WORK)/full.log \
		| sed -n 's|^- package github.com/tmc/lib/nvidia/\([^:]*\): added$$|  \1|p' \
		| sort

clean:
	git -C $(REPO) worktree remove --force $(WORK)/tag 2>/dev/null || true
	rm -rf $(WORK)
