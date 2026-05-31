#!/bin/bash
BASE=/projects/slmreasoning/junh/causal-error/CausalError/baselines/outputs

FOLDERS=(
    "$BASE/outputs_-common-data-models-openai--gpt-oss-120b-GAIA_dedup-who_and_when_w1"
    "$BASE/outputs_-common-data-models-openai--gpt-oss-120b-SWE_Bench_dedup-who_and_when_w1"
    "$BASE/outputs_Tongyi-Zhiwen-QwenLong-L1-32B-GAIA_dedup-who_and_when_w1"
)

for folder in "${FOLDERS[@]}"; do
    for f in "$folder"/*.json; do
        [[ "$f" == *"_meta_"* ]] && continue
        grep -q '"_error": "json_parse_failed"' "$f" || continue
        trace=$(basename "$f" .json)
        rm "$f"
        rm -f "$folder/_meta_${trace}.json"
        echo "deleted: $(basename $folder)/$(basename $f)"
    done
done

echo "Done."
