#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# wetspring_composition.sh — Data Exploration & Visualization Composition
#
# Explores the wetSpring lane: genome/protein visualization via petalTongue
# scene graphs, DAG-linked biological data navigation via rhizoCrypt,
# large dataset handling via NestGate storage, and braid lineage for
# scientific data provenance via sweetGrass.
#
# Built from primalSpring Phase 46 composition_template.sh.
# See primalSpring/wateringHole/DOWNSTREAM_COMPOSITION_EXPLORER_GUIDE.md
#
# Usage:
#   COMPOSITION_NAME=wetspring ./tools/wetspring_composition.sh
#   COMPOSITION_NAME=wetspring FAMILY_ID=wetspring ./tools/wetspring_composition.sh

set -euo pipefail

# ── 1. Configuration ──────────────────────────────────────────────────

COMPOSITION_NAME="${COMPOSITION_NAME:-wetspring}"
REQUIRED_CAPS="visualization security tensor"
OPTIONAL_CAPS="compute dag ledger attribution ai storage"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/nucleus_composition_lib.sh"

# ── 2. Domain State — Biological Dataset ──────────────────────────────
#
# A small protein/gene dataset for exploration. In production this would
# come from SRA/UniProt/PDB accessions; here we use a curated subset
# from wetSpring's Exp301 (protein folding) and Exp310 (phylogenetics).

RUNNING=true
SELECTED_IDX=-1
DETAIL_MODE=false
EXPLORATION_DEPTH=0
ACCUMULATED_HOVER_MOVES=0

declare -a GENE_IDS=("BRCA1" "TP53" "EGFR" "KRAS" "MYC" "PTEN" "AKT1" "PIK3CA")
declare -a GENE_NAMES=(
    "Breast cancer type 1"
    "Tumor protein p53"
    "Epidermal growth factor receptor"
    "GTPase KRas"
    "Myc proto-oncogene"
    "Phosphatase and tensin homolog"
    "AKT serine/threonine kinase 1"
    "PI3-kinase catalytic subunit alpha"
)
declare -a GENE_SEQS=(
    "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLKLLNQKKGPSQCPLC"
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDE"
    "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLG"
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSHKRKIQDMRAYRKLRQDALIPYIE"
    "MDFFRVVENQQPATMPLNVSFTNRNYDLDYDSVQPYFYCDEEENFYQQQQQSELQPPAPSEDI"
    "MTAIIKEIVSRNKRRYQEDGFDLDLTYIYPNIIAMGFPAERLEGVYRNNIDDVVRFLDSKHKN"
    "MSDVAIVKEGWLHKRGEYIKTWRPRYFLLKNDGTFIGYKERPQDVDQREAPLNNFSVAQCQLM"
    "MPPRPSSGELWGIHLMPPRILVECLLPNGMIVTLECLREATLITIKHELFKEARKYPLHQLLQD"
)

declare -A GENE_LINKS=(
    [BRCA1]="TP53 AKT1 PTEN"
    [TP53]="BRCA1 MYC KRAS PTEN"
    [EGFR]="KRAS PIK3CA AKT1"
    [KRAS]="TP53 EGFR MYC PIK3CA"
    [MYC]="TP53 KRAS"
    [PTEN]="BRCA1 TP53 AKT1 PIK3CA"
    [AKT1]="BRCA1 EGFR PTEN PIK3CA"
    [PIK3CA]="EGFR KRAS PTEN AKT1"
)

# ── 3. Hit Testing ───────────────────────────────────────────────────
#
# Layout: gene list on the left (8 rows, 60px each starting at y=120)
# Detail panel on the right when a gene is selected.

GENE_ROW_HEIGHT=60
GENE_LIST_X=40
GENE_LIST_Y=120
GENE_LIST_WIDTH=300

hit_test_fn() {
    local px="$1" py="$2"
    px="${px%.*}"
    py="${py%.*}"

    if (( px >= GENE_LIST_X && px < GENE_LIST_X + GENE_LIST_WIDTH )); then
        local row=$(( (py - GENE_LIST_Y) / GENE_ROW_HEIGHT ))
        if (( row >= 0 && row < ${#GENE_IDS[@]} )); then
            echo "$row"
            return
        fi
    fi
    echo -1
}

# ── 4. Domain Hooks ──────────────────────────────────────────────────

domain_init() {
    dag_create_session "wetspring-exploration" \
        "[{\"key\":\"domain\",\"value\":\"life_science\"},{\"key\":\"dataset\",\"value\":\"cancer_gene_panel_8\"}]"
    ledger_create_spine

    if cap_available storage; then
        store_dataset_to_nestgate
    fi

    domain_render "Ready — click a gene or press arrow keys to navigate"
}

store_dataset_to_nestgate() {
    log "storing dataset to NestGate (testing large-payload IPC)..."
    local storage_sock
    storage_sock=$(cap_socket storage) || return

    local dataset_json='{"genes":['
    for i in "${!GENE_IDS[@]}"; do
        [[ $i -gt 0 ]] && dataset_json+=","
        dataset_json+="{\"id\":\"${GENE_IDS[$i]}\",\"name\":\"${GENE_NAMES[$i]}\",\"seq_len\":${#GENE_SEQS[$i]}}"
    done
    dataset_json+='],"panel":"cancer_gene_panel_8","source":"wetSpring_Exp301_Exp310","accessions":["BRCA1:P38398","TP53:P04637","EGFR:P00533","KRAS:P01116"]}'

    local resp
    resp=$(send_rpc "$storage_sock" "storage.store" \
        "{\"key\":\"wetspring-gene-panel\",\"value\":$dataset_json}")
    if [[ -n "$resp" ]]; then
        log "NestGate store: $(echo "$resp" | head -c 120)"
    else
        log "NestGate store: no response (PG-04 / absent)"
    fi

    resp=$(send_rpc "$storage_sock" "storage.retrieve" \
        "{\"key\":\"wetspring-gene-panel\"}")
    if [[ -n "$resp" ]]; then
        local len=${#resp}
        log "NestGate retrieve: ${len} bytes returned (large-payload IPC test)"
        braid_record "storage_test" "application/x-wetspring" "dataset_loaded" \
            "{\"key\":\"wetspring-gene-panel\",\"response_bytes\":$len}" "system" "0"
    else
        log "NestGate retrieve: no response"
    fi
}

domain_render() {
    local status="${1:-}"

    local title
    title=$(make_text_node "title" 300 40 "wetSpring — Gene Explorer" 28 0.4 0.85 0.65)
    local subtitle
    subtitle=$(make_text_node "subtitle" 300 70 "$status" 14 0.7 0.7 0.75)

    local children='"title","subtitle"'
    local nodes="${title},${subtitle}"

    for i in "${!GENE_IDS[@]}"; do
        local gene_id="${GENE_IDS[$i]}"
        local gene_name="${GENE_NAMES[$i]}"
        local ypos=$(( GENE_LIST_Y + i * GENE_ROW_HEIGHT ))
        local r=0.75 g=0.75 b=0.8
        local size=16

        if (( i == SELECTED_IDX )); then
            r=0.3; g=0.95; b=0.6; size=18
        fi

        local node_id="gene_${i}"
        local label="${gene_id} — ${gene_name}"
        local node
        node=$(make_text_node "$node_id" $((GENE_LIST_X + 10)) "$ypos" "$label" "$size" "$r" "$g" "$b")
        nodes="${nodes},${node}"
        children="${children},\"${node_id}\""
    done

    if $DETAIL_MODE && (( SELECTED_IDX >= 0 )); then
        local sel_gene="${GENE_IDS[$SELECTED_IDX]}"
        local sel_name="${GENE_NAMES[$SELECTED_IDX]}"
        local sel_seq="${GENE_SEQS[$SELECTED_IDX]}"
        local links="${GENE_LINKS[$sel_gene]:-none}"

        local detail_x=420
        local dheader
        dheader=$(make_text_node "detail_header" $detail_x 130 "[$sel_gene] $sel_name" 20 0.3 0.95 0.6)
        nodes="${nodes},${dheader}"
        children="${children},\"detail_header\""

        local seq_display="${sel_seq:0:40}"
        [[ ${#sel_seq} -gt 40 ]] && seq_display="${seq_display}..."
        local dseq
        dseq=$(make_text_node "detail_seq" $detail_x 165 "Seq: $seq_display" 12 0.6 0.65 0.7)
        nodes="${nodes},${dseq}"
        children="${children},\"detail_seq\""

        local dlen
        dlen=$(make_text_node "detail_len" $detail_x 190 "Length: ${#sel_seq} aa | Links: $links" 12 0.6 0.65 0.7)
        nodes="${nodes},${dlen}"
        children="${children},\"detail_len\""

        local ddepth
        ddepth=$(make_text_node "detail_depth" $detail_x 215 "DAG depth: $EXPLORATION_DEPTH | Braids: \${LAST_BRAID_ID:-0}" 12 0.5 0.55 0.6)
        nodes="${nodes},${ddepth}"
        children="${children},\"detail_depth\""

        local link_y=250
        local link_idx=0
        for linked_gene in $links; do
            local lnode
            lnode=$(make_text_node "link_${link_idx}" $detail_x $link_y "→ $linked_gene" 14 0.5 0.7 0.9)
            nodes="${nodes},${lnode}"
            children="${children},\"link_${link_idx}\""
            link_y=$((link_y + 30))
            link_idx=$((link_idx + 1))
        done
    fi

    local provenance_y=500
    local prov_node
    prov_node=$(make_text_node "provenance" 40 $provenance_y "Exploration: depth=$EXPLORATION_DEPTH | vertex=${CURRENT_VERTEX:-genesis}" 11 0.45 0.45 0.5)
    nodes="${nodes},${prov_node}"
    children="${children},\"provenance\""

    local root
    root=$(printf '"root":{"id":"root","transform":{"a":1.0,"b":0.0,"c":0.0,"d":1.0,"tx":0.0,"ty":0.0},"primitives":[],"children":[%s],"visible":true,"opacity":1.0,"label":"wetspring-explorer","data_source":"cancer_gene_panel_8"}' "$children")
    local scene="{\"nodes\":{${root},${nodes}},\"root_id\":\"root\"}"
    push_scene "${COMPOSITION_NAME}-main" "$scene"
}

domain_on_key() {
    local key="$1"
    case "$key" in
        Q|q|Escape)
            log "quit requested"
            RUNNING=false
            ;;
        Up|k)
            if (( SELECTED_IDX > 0 )); then
                SELECTED_IDX=$((SELECTED_IDX - 1))
                navigate_to_gene "$SELECTED_IDX" "keyboard"
            fi
            ;;
        Down|j)
            if (( SELECTED_IDX < ${#GENE_IDS[@]} - 1 )); then
                SELECTED_IDX=$((SELECTED_IDX + 1))
                navigate_to_gene "$SELECTED_IDX" "keyboard"
            fi
            ;;
        Return|Enter|space)
            if (( SELECTED_IDX >= 0 )); then
                DETAIL_MODE=true
                log "detail view for ${GENE_IDS[$SELECTED_IDX]}"
                domain_render "Detail: ${GENE_IDS[$SELECTED_IDX]}"
            fi
            ;;
        BackSpace|b)
            if $DETAIL_MODE; then
                DETAIL_MODE=false
                domain_render "List view (depth=$EXPLORATION_DEPTH)"
            fi
            ;;
        s)
            test_storage_roundtrip
            ;;
        p)
            query_provenance
            ;;
        m)
            test_math_ipc
            ;;
        *)
            log "unhandled key: $key"
            dag_append_event "wetspring" "keypress" "nav:${SELECTED_IDX}" \
                "[{\"key\":\"key\",\"value\":\"$key\"}]" "keyboard" "0"
            ;;
    esac
}

navigate_to_gene() {
    local idx="$1" input_type="$2"
    local gene_id="${GENE_IDS[$idx]}"
    EXPLORATION_DEPTH=$((EXPLORATION_DEPTH + 1))

    dag_append_event "wetspring" "navigate" "gene:$gene_id" \
        "[{\"key\":\"gene\",\"value\":\"$gene_id\"},{\"key\":\"index\",\"value\":\"$idx\"}]" \
        "$input_type" "$ACCUMULATED_HOVER_MOVES"

    braid_record "navigate" "application/x-wetspring-gene" "gene:$gene_id" \
        "{\"gene\":\"$gene_id\",\"name\":\"${GENE_NAMES[$idx]}\",\"seq_len\":${#GENE_SEQS[$idx]},\"depth\":$EXPLORATION_DEPTH}" \
        "$input_type" "$ACCUMULATED_HOVER_MOVES"

    ACCUMULATED_HOVER_MOVES=0
    domain_render "Selected: $gene_id (depth=$EXPLORATION_DEPTH)"
}

domain_on_click() {
    local cell="$1"
    if (( cell >= 0 && cell < ${#GENE_IDS[@]} )); then
        SELECTED_IDX=$cell
        DETAIL_MODE=true
        navigate_to_gene "$cell" "click"
    fi
}

domain_on_tick() {
    check_proprioception
}

# ── 5. Exploration Probes ─────────────────────────────────────────────
#
# Additional IPC probes that exercise the composition beyond basic
# navigation. Triggered by hotkeys during interactive exploration.

test_storage_roundtrip() {
    if ! cap_available storage; then
        log "storage capability not available (skip)"
        domain_render "Storage: not available"
        return
    fi
    local storage_sock
    storage_sock=$(cap_socket storage)
    local test_key="wetspring-probe-$(date +%s)"
    local payload='{"type":"sequence_fragment","data":"ATCGATCGATCGATCGATCGATCGATCGATCGATCG","organism":"E.coli","accession":"NC_000913.3","region":"16S rRNA partial"}'
    local resp
    resp=$(send_rpc "$storage_sock" "storage.store" "{\"key\":\"$test_key\",\"value\":$payload}")
    log "storage.store probe: $(echo "$resp" | head -c 80)"

    resp=$(send_rpc "$storage_sock" "storage.retrieve" "{\"key\":\"$test_key\"}")
    local len=${#resp}
    log "storage.retrieve probe: $len bytes"

    braid_record "storage_probe" "application/x-wetspring" "roundtrip" \
        "{\"key\":\"$test_key\",\"store_ok\":true,\"retrieve_bytes\":$len}" "system" "0"
    domain_render "Storage roundtrip: ${len}B returned (key=$test_key)"
}

query_provenance() {
    log "querying provenance tree..."
    local tree
    tree=$(braid_provenance_tree)
    log "provenance tree: $(echo "$tree" | head -c 200)"

    local recent
    recent=$(braid_query_recent 5)
    log "recent braids: $(echo "$recent" | head -c 200)"

    domain_render "Provenance: tree queried, ${EXPLORATION_DEPTH} navigation steps"
}

test_math_ipc() {
    if ! cap_available tensor; then
        log "tensor capability not available (skip)"
        domain_render "Math IPC: tensor not available"
        return
    fi
    local tensor_sock
    tensor_sock=$(cap_socket tensor)

    log "testing barraCuda IPC: stats.mean on gene sequence lengths..."
    local lengths="["
    for i in "${!GENE_SEQS[@]}"; do
        [[ $i -gt 0 ]] && lengths+=","
        lengths+="${#GENE_SEQS[$i]}.0"
    done
    lengths+="]"

    local resp
    resp=$(send_rpc "$tensor_sock" "stats.mean" "{\"data\":$lengths}")
    log "stats.mean(seq_lengths) = $(echo "$resp" | head -c 120)"

    resp=$(send_rpc "$tensor_sock" "stats.std_dev" "{\"data\":$lengths}")
    log "stats.std_dev(seq_lengths) = $(echo "$resp" | head -c 120)"

    resp=$(send_rpc "$tensor_sock" "stats.correlation" \
        "{\"x\":$lengths,\"y\":$lengths}")
    log "stats.correlation(self) = $(echo "$resp" | head -c 120)"

    braid_record "math_probe" "application/x-wetspring" "ipc_test" \
        "{\"methods\":[\"stats.mean\",\"stats.std_dev\",\"stats.correlation\"],\"n_genes\":${#GENE_IDS[@]}}" \
        "system" "0"
    domain_render "Math IPC: 3 barraCuda methods exercised on ${#GENE_IDS[@]} gene lengths"
}

# ── 6. Main Loop ─────────────────────────────────────────────────────

main() {
    discover_capabilities || { err "Required primals not found"; exit 1; }

    composition_startup "wetSpring Gene Explorer" "Data Exploration & Visualization — Phase 46"

    subscribe_interactions "click"
    subscribe_sensor_stream

    domain_init

    log "hotkeys: Up/Down=navigate  Enter=detail  b=back  s=storage  p=provenance  m=math  q=quit"

    while $RUNNING; do
        local sensor_batch
        sensor_batch=$(poll_sensor_stream)
        process_sensor_batch "$sensor_batch"

        ACCUMULATED_HOVER_MOVES=$((ACCUMULATED_HOVER_MOVES + SENSOR_HOVER_MOVES))

        if $SENSOR_HOVER_CHANGED; then
            domain_render "Hovering gene ${HOVER_CELL} (depth=$EXPLORATION_DEPTH)"
        fi

        if [[ -n "$SENSOR_KEY" ]]; then
            domain_on_key "$SENSOR_KEY"
        elif [[ "$SENSOR_CLICK_CELL" -ge 0 ]]; then
            domain_on_click "$SENSOR_CLICK_CELL"
        else
            domain_on_tick
            sleep "$POLL_INTERVAL"
        fi
    done

    log "exploration complete: $EXPLORATION_DEPTH navigation steps"
    composition_summary
    composition_teardown "${COMPOSITION_NAME}-main"
}

main
