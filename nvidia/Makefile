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

.PHONY: regen version-check api-update clean

# Regenerate every binding package, then check what the regeneration did to
# the API. The check runs here rather than in a person's memory: a rename
# that leaves constant values alone is invisible to every other gate.
regen:
	sh regen.sh
	$(MAKE) version-check

# Fail if the module's incompatible-change set against $(TAG) differs from
# the committed report. An entry is reviewed when it is in that report.
#
# Both sides are sorted before comparing. apidiff emits its entries in an
# order that is not stable across runs or platforms, so an unsorted diff
# reports a difference on every invocation. Sorting separates the warning's
# continuation lines from their entry, which makes the diff a set difference
# and not a readable report; read api/nvidia.log itself for that.
version-check: $(WORK)/old.api $(WORK)/new.api
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

clean:
	git -C $(REPO) worktree remove --force $(WORK)/tag 2>/dev/null || true
	rm -rf $(WORK)
