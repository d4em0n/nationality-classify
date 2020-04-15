# Nationality Classify
This CNN model classified country based on person name.

# Avaliable Nationality
This model trained with [140317 names](./train.csv) from 18 nationality/ethnics:
- Russian
- China
- Arabic
- Dutch
- Korean
- Polish
- Scottish
- Italian
- UK
- France
- Japan
- Greece
- Spanish
- India
- Turkish
- Indonesia
- Vietnam

# Result
For testing, we use 35080 mix names from available countries. Using `evaluation.py` we get 96.22% accuracy

# Confusion Matrix
![Confusion Matrix](./conf_matrix.png)
