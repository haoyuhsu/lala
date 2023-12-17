# collect demos for stack-towers
python cliport/demos.py \
    n=500 \
    task=stack-towers \
    mode=train

python cliport/demos.py \
    n=100 \
    task=stack-towers \
    mode=val 

# collect demos for stack-boxes
python cliport/demos.py \
    n=500 \
    task=stack-boxes \
    mode=train

python cliport/demos.py \
    n=100 \
    task=stack-boxes \
    mode=val