# Demo script that reproduces the experimental results of our work.
# Please set --epochs 1 --iters 1 if you want to check whether the code runs.

cd src

echo "Checking data..."
python data.py

echo "No augmentation:"
python run.py --data all --batch-size 32 128 --dropout 0 0.5

echo "NodeSam:"
python run.py --data all --batch-size 32 128 --dropout 0 0.5 --augment NodeSam

echo "SubMix:"
python run.py --data all --batch-size 32 128 --dropout 0 0.5 --augment SubMix
