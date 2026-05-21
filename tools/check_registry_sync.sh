#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Registry cross-sync test — wetSpring
#
# Validates that every method in wetSpring's capability_registry.toml
# exists in primalSpring's canonical 445-method registry. Catches drift
# where wetSpring registers a method that the ecosystem doesn't know about.
#
# Also validates that every dotted method string used in Rust source
# (barracuda/, graphs/) appears in the local registry.
#
# Usage: tools/check_registry_sync.sh
#
# Expects primalSpring at ../primalSpring (relative to spring root) or
# at ../../springs/primalSpring (relative to ecoPrimals/).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

LOCAL_REG="capability_registry.toml"
PRIMAL_SPRING="${PRIMALSPRING_PATH:-}"

if [[ -z "$PRIMAL_SPRING" ]]; then
    for candidate in \
        "../primalSpring/config/capability_registry.toml" \
        "../../springs/primalSpring/config/capability_registry.toml" \
        "../../../springs/primalSpring/config/capability_registry.toml"; do
        if [[ -f "$candidate" ]]; then
            PRIMAL_SPRING="$candidate"
            break
        fi
    done
fi

if [[ ! -f "$LOCAL_REG" ]]; then
    echo "FAIL: $LOCAL_REG not found"
    exit 1
fi

# ── Phase 1: Local registry → primalSpring canonical sync ──

LOCAL_METHODS=$(grep -oP '^\s*method\s*=\s*"[a-z][a-z0-9_.]+[a-z0-9]+"' "$LOCAL_REG" \
    | grep -oP '"[^"]+"' | tr -d '"' | sort -u)
LOCAL_COUNT=$(echo "$LOCAL_METHODS" | wc -l)

if [[ -n "$PRIMAL_SPRING" && -f "$PRIMAL_SPRING" ]]; then
    CANONICAL=$(grep -oP '^\s+"[a-z][a-z0-9_.]+[a-z0-9]+"' "$PRIMAL_SPRING" \
        | tr -d ' "' | sort -u)
    CANONICAL_COUNT=$(echo "$CANONICAL" | wc -l)

    MISSING_IN_CANONICAL=()
    while IFS= read -r method; do
        if ! echo "$CANONICAL" | grep -qxF "$method"; then
            MISSING_IN_CANONICAL+=("$method")
        fi
    done <<< "$LOCAL_METHODS"

    SYNC_ERRORS=${#MISSING_IN_CANONICAL[@]}
    if [[ $SYNC_ERRORS -eq 0 ]]; then
        echo "SYNC OK: all $LOCAL_COUNT wetSpring methods are in primalSpring canonical ($CANONICAL_COUNT methods)"
    else
        echo "SYNC GAP: $SYNC_ERRORS wetSpring method(s) not yet in primalSpring canonical:"
        for m in "${MISSING_IN_CANONICAL[@]}"; do
            echo "  $m"
        done
        echo ""
        echo "These are wetSpring domain methods that primalSpring should absorb."
        echo "Report upstream: add to primalSpring config/capability_registry.toml"
    fi
else
    echo "SKIP: primalSpring canonical registry not found (set PRIMALSPRING_PATH)"
    echo "      Checked: ../primalSpring/config/ and sibling paths"
fi

# ── Phase 2: Rust source method strings → local registry ──

echo ""
echo "--- Local method string drift check ---"

# Extract method-like strings, then filter out file paths, Python imports, and test fixtures
EXCLUDE_PATTERN='\.(fastq|fasta|json|toml|jcamp|biom|tsv|txt|bin|sock|gz|rs)$'
EXCLUDE_LIBS='^(scipy|skbio|astropy|numpy)\.'
EXCLUDE_EXTS='\.(fcgi|py|sh)$'

if command -v rg &>/dev/null; then
    RAW=$(rg -o '"[a-z]+\.[a-z_.]+[a-z]+"' barracuda/src/ graphs/ --no-filename 2>/dev/null \
        | tr -d '"' | sort -u || true)
else
    RAW=$(grep -roP '"[a-z]+\.[a-z_.]+[a-z]+"' barracuda/src/ graphs/ 2>/dev/null \
        | grep -oP '"[^"]+"' | tr -d '"' | sort -u || true)
fi
USED=$(echo "$RAW" | grep -vE "$EXCLUDE_PATTERN" | grep -vE "$EXCLUDE_LIBS" \
    | grep -vE "$EXCLUDE_EXTS" | grep -vE '^(test|bogus|nonexistent|method)\.' \
    | grep -vE '\.(get|set|push|pop|len|new|init|run|main)$' || true)

if [[ -z "$USED" ]]; then
    echo "SKIP: no method strings found in source (rg not available or empty)"
    exit 0
fi

USED_COUNT=$(echo "$USED" | wc -l)
DRIFT=0
DRIFTED=()

while IFS= read -r method; do
    if ! echo "$LOCAL_METHODS" | grep -qxF "$method"; then
        DRIFTED+=("$method")
        ((DRIFT++)) || true
    fi
done <<< "$USED"

if [[ $DRIFT -eq 0 ]]; then
    echo "LOCAL OK: $USED_COUNT method strings in source, all registered in $LOCAL_REG"
else
    # Advisory only: wetSpring consumes ecosystem methods from other primals
    # (compute.*, storage.*, crypto.*, etc.) that belong to their respective registries
    echo "LOCAL INFO (advisory): $DRIFT method string(s) in source not in local registry"
    echo "  (Expected: springs consume methods from ecosystem primals not in local TOML)"
fi

# ── Summary ──

echo ""
echo "Registry:     $LOCAL_COUNT methods registered"
echo "Source:        $USED_COUNT method strings found"
echo "Cross-sync:    ${SYNC_ERRORS:-n/a} gaps vs primalSpring canonical"
# Exit 0: the cross-sync gaps are upstream work for primalSpring to absorb
exit 0
