for i in {0..100..20}
do
    head -$i runs.txt | tail -20 | parallel python3.8 empiric_to_person_object.py
done