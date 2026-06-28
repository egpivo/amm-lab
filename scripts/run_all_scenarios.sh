#!/usr/bin/env bash
set -euo pipefail

BINARY="cargo run --release --"
OUT_DIR="data/processed"

echo "=== amm-lab: run all article scenarios ==="
echo "Output directory: ${OUT_DIR}"
echo ""

echo "--- [1/4] price_impact_ladder ---"
set -x
${BINARY} scenario run scenarios/price_impact_ladder.toml
{ set +x; } 2>/dev/null

echo ""
echo "--- [2/4] same_price_different_depth ---"
set -x
${BINARY} scenario run scenarios/same_price_different_depth.toml
{ set +x; } 2>/dev/null

echo ""
echo "--- [3/4] arbitrage_repricing ---"
set -x
${BINARY} scenario run scenarios/arbitrage_repricing.toml
{ set +x; } 2>/dev/null

echo ""
echo "--- [4/4] lp_vs_hold ---"
set -x
${BINARY} scenario run scenarios/lp_vs_hold.toml
{ set +x; } 2>/dev/null

echo ""
echo "=== All scenarios complete ==="
echo ""
echo "Artifacts written to: ${OUT_DIR}/"
echo ""
echo "Next steps if article numbers changed:"
echo "  1. Update data/manifests/artifact_manifest.json  (sha256 each file)"
echo "  2. Update research/evidence_ledger.csv           (re-check field values)"
echo "  3. Update .local/research/publication_snapshot.json"
