for i in {0..100..10}
do
    head -$i runs.txt | tail -10 | parallel python3.8 disease_transmission.py
done