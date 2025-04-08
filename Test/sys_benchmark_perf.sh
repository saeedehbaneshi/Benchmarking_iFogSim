#!/bin/bash

OUTPUT_FILE="perf_results.txt"
NUM_RUNS=10

# Create/overwrite CSV with header:
echo "Run,Cycles,Instructions,CacheRefs,CacheMisses,BranchIns,BranchMisses,EnergyPkg(J),EnergyCores(J),EnergyPsys(J),ElapsedTime(s),MIPS,IPC,IPS" > "$OUTPUT_FILE"

for i in $(seq 1 "$NUM_RUNS"); do
    echo "Starting run $i..."

    # 1) Run perf in system-wide mode with NO columns to avoid wrapping:
    sudo perf stat -a -e \
      cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses,\
power/energy-pkg/,power/energy-cores/,power/energy-psys/ \
      -o temp_perf.txt \
      /media/saeedeh/Data/PhD/Benchmarking/env_yolo/bin/python3 Yolov3_inference.py 2>&1

    # 2) Parse key metrics from the plain-text summary

    # Remove commas from large numbers so 'bc' can parse them
    CYCLES=$(grep "cycles" temp_perf.txt | awk '{print $1}' | tr -d ',')
    INSTRUCTIONS=$(grep "instructions" temp_perf.txt | awk '{print $1}' | tr -d ',')
    CACHE_REF=$(grep "cache-references" temp_perf.txt | awk '{print $1}' | tr -d ',')
    CACHE_MISS=$(grep "cache-misses" temp_perf.txt | awk '{print $1}' | tr -d ',')
    BRANCH_INS=$(grep "branch-instructions" temp_perf.txt | awk '{print $1}' | tr -d ',')
    BRANCH_MISS=$(grep "branch-misses" temp_perf.txt | awk '{print $1}' | tr -d ',')

    ENERGY_PKG=$(grep "power/energy-pkg/" temp_perf.txt | awk '{print $1}')
    ENERGY_CORES=$(grep "power/energy-cores/" temp_perf.txt | awk '{print $1}')
    ENERGY_PSYS=$(grep "power/energy-psys/" temp_perf.txt | awk '{print $1}')

    ELAPSED_TIME=$(grep "seconds time elapsed" temp_perf.txt | awk '{print $1}')

    # 3) Calculate derived metrics

    # MIPS (Millions of Instructions Per Second)
    # = (instructions / 1e6) / elapsed_time
    if [[ -n "$INSTRUCTIONS" && -n "$ELAPSED_TIME" && "$ELAPSED_TIME" != "0" ]]; then
        MIPS=$(echo "scale=2; $INSTRUCTIONS / 1000000 / $ELAPSED_TIME" | bc)
    else
        MIPS="N/A"
    fi

    # IPC (Instructions Per Cycle)
    if [[ -n "$INSTRUCTIONS" && -n "$CYCLES" && "$CYCLES" != "0" ]]; then
        IPC=$(echo "scale=2; $INSTRUCTIONS / $CYCLES" | bc)
    else
        IPC="N/A"
    fi

    # IPS (Instructions Per Second)
    # = instructions / elapsed_time
    if [[ -n "$INSTRUCTIONS" && -n "$ELAPSED_TIME" && "$ELAPSED_TIME" != "0" ]]; then
        IPS=$(echo "scale=2; $INSTRUCTIONS / $ELAPSED_TIME" | bc)
    else
        IPS="N/A"
    fi

    # 4) Append one CSV line for this run
    echo "$i,$CYCLES,$INSTRUCTIONS,$CACHE_REF,$CACHE_MISS,$BRANCH_INS,$BRANCH_MISS,$ENERGY_PKG,$ENERGY_CORES,$ENERGY_PSYS,$ELAPSED_TIME,$MIPS,$IPC,$IPS" >> "$OUTPUT_FILE"

done

# Clean up
rm temp_perf.txt
echo "Done. Results saved to '$OUTPUT_FILE'."

