for i in $(seq 0 99)
do
    for method in "Retrace" "Moretrace" "TruncatedIS"
    do
        # Don't forget & after command here to run in parallel
        python trace_decay.py $method $i > "data/${method}_$i.txt" &
    done
done
