for i in {0..100..20}
do
    head -$i runs.txt | tail -20 | parallel python3.8 disease_transmission.py
done