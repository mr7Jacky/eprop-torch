for (( l=5 ; l<=10 ; l++ )); 
do
    for n in {1..50..5}; 
    do  
        res=`echo "scale=2; $n / 10" | bc`
        echo "--threshold $res --epochs 1 --lr 1e-$l"
        python main.py --threshold $res --epochs 1 --lr 1e-$l
    done
done