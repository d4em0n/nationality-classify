# Hello CNN!
This is my first experiment with deep learning. this CNN model classified country based on person name, this CNN trained using chinese, russian, arabic and germany.

# Datasets
- There are 9800 chinese, russian, arabic and germany name (total 39200 person's name)
- This model trained with 31360 random mixed chinese, russian and arabic name the rest of 7840 is used for evaluation/testing data.
- train.csv and evaluation.csv extracted using gen_data.py (for parsing *dataset.txt file and seperate train and testing data)

# Evaluation result
Using `evaluation.py` get the output:
```
accuracy: 99.11%
```

# single_evaluation.py
script single_evaluation.py is for evaluation single person name get from the argument
```
$ python single_evaluation.py
output:
Enter the name you want to classify: nur kholifah
probability russian names = 0.03%
probability chinese names = 0.00%
probability arabic names = 99.97%
probability germany names = 0.00%
Enter the name you want to classify: xi jinping
probability russian names = 0.00%
probability chinese names = 100.00%
probability arabic names = 0.00%
probability germany names = 0.00%
Enter the name you want to classify: vladimir putin
probability russian names = 100.00%
probability chinese names = 0.00%
probability arabic names = 0.00%
probability germany names = 0.00%
Enter the name you want to classify: muhammad alifa ramdhan
probability russian names = 0.40%
probability chinese names = 0.00%
probability arabic names = 99.60%
probability germany names = 0.00%
Enter the name you want to classify: Frank-Walter Steinmeier
probability russian names = 0.00%
probability chinese names = 0.00%
probability arabic names = 0.00%
probability germany names = 100.00%
```
