import os

print("Running preprocessing...")
os.system("python src/preprocess.py")

print("Training model...")
os.system("python src/train.py")

print("Evaluating model...")
os.system("python src/evaluate.py")

print("Pipeline completed!")