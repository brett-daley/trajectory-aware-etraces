for i in $(seq 0 99)
do
    # Don't forget & after command here to run in parallel
    python trace_decay.py Retrace 0.4 $i > "data/Retrace_$i.txt" &
    python trace_decay.py Moretrace 0.45 $i > "data/Moretrace_$i.txt" &
    python trace_decay.py RecursiveRetrace 0.05 $i > "data/RecursiveRetrace_$i.txt" &
    python trace_decay.py TruncatedIS 0.4 $i > "data/TruncatedIS_$i.txt" &
done
