#!/bin/bash

OUTPUT_FILE="perf_results.txt"
NUM_RUNS=10

# Create (or overwrite) CSV with a header row:
echo "Run,Cycles,Instructions,CacheRefs,CacheMisses,BranchIns,BranchMisses,ElapsedTime(s),UserTime(s),SysTime(s),MIPS,IPC,IPS" > "$OUTPUT_FILE"

for i in $(seq 1 "$NUM_RUNS"); do
    echo "Starting run $i..."
    
    export COLUMNS=2000
    export PERF_PAGER=cat

    # --- PERF COMMAND ---
    # Per-process measurement (no '-a'), collecting basic events:
    perf stat \
      -e cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses \
      -o temp_perf.txt \
      /media/saeedeh/Data/PhD/Benchmarking/env_yolo/bin/python3 Yolov3_inference.py 2>&1

    # --- PARSE THE RESULTS ---
    # Each line in 'temp_perf.txt' looks like:
    #   <number> <event_name> ... 
    # We grep each event name and extract the numeric field, removing commas.

    CYCLES=$(grep "cycles" temp_perf.txt | awk '{print $1}' | tr -d ',')
    INSTRUCTIONS=$(grep "instructions" temp_perf.txt | awk '{print $1}' | tr -d ',')
    CACHE_REF=$(grep "cache-references" temp_perf.txt | awk '{print $1}' | tr -d ',')
    CACHE_MISS=$(grep "cache-misses" temp_perf.txt | awk '{print $1}' | tr -d ',')
    BRANCH_INS=$(grep "branch-instructions" temp_perf.txt | awk '{print $1}' | tr -d ',')
    BRANCH_MISS=$(grep "branch-misses" temp_perf.txt | awk '{print $1}' | tr -d ',')

    # These lines appear in the default perf summary for a per-process measurement:
    #   X.XXXXX seconds time elapsed
    #   X.XXXXX seconds user
    #   X.XXXXX seconds sys
    ELAPSED_TIME=$(grep "seconds time elapsed" temp_perf.txt | awk '{print $1}')
    USER_TIME=$(grep "seconds user" temp_perf.txt | awk '{print $1}')
    SYS_TIME=$(grep "seconds sys" temp_perf.txt | awk '{print $1}')

    # --- DERIVED METRICS ---
    # MIPS (Millions of Instructions Per Second)
    # = (Instructions / 1e6) / ElapsedTime
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
    # = Instructions / ElapsedTime
    if [[ -n "$INSTRUCTIONS" && -n "$ELAPSED_TIME" && "$ELAPSED_TIME" != "0" ]]; then
        IPS=$(echo "scale=2; $INSTRUCTIONS / $ELAPSED_TIME" | bc)
    else
        IPS="N/A"
    fi

    # --- APPEND CSV ROW ---
    echo "$i,$CYCLES,$INSTRUCTIONS,$CACHE_REF,$CACHE_MISS,$BRANCH_INS,$BRANCH_MISS,$ELAPSED_TIME,$USER_TIME,$SYS_TIME,$MIPS,$IPC,$IPS" >> "$OUTPUT_FILE"

    # Clean up
    rm -f temp_perf.txt

done

echo "Done. Results saved to '$OUTPUT_FILE'."

