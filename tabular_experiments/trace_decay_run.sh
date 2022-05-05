N=24  # Parallel batch size
for i in $(seq 0 25)
do
    for method in "Retrace" "Moretrace" "IS"
    do
        # Run in parallel batches of N
        ((i=i%N)); ((i==0)) && wait
        # Don't forget & after command here to run in parallel
        python trace_decay.py $method $i > "data/${method}_$i.txt" &
    done
done
