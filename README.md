<<<<<<< HEAD
# Mushroom Classification using Decision Tree

## Overview
This project implements a decision tree classifier to identify edible and poisonous mushrooms based on their physical characteristics. The model achieves 98.87% accuracy using a pruned decision tree with only 23 nodes.

## Key Features
- Data preprocessing and analysis
- Decision tree model with cost complexity pruning
- Comprehensive model evaluation
- Feature importance analysis
- Detailed visualization of decision rules

## Model Performance
- Accuracy: 98.87%
- Precision (Poisonous): 98%
- Recall (Poisonous): 100%
- Precision (Edible): 100%
- Recall (Edible): 98%

## Most Important Features
1. Gill Color (35.73%)
2. Spore Print Color (19.68%)
3. Population (18.50%)

## Files Description
- `decisiontree_generic.py`: Main script for training the model
- `decisiontree.ipynb`: Jupyter notebook containing detailed analysis
- `Mushrooms.csv`: Dataset containing mushroom characteristics

## Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Usage
1. Run the generic script:
```bash
python decisiontree_generic.py --csv "Mushrooms.csv" --target class --map e:0,p:1 --max_depth 6 --min_leaf 10 --outname mushroom
```

2. Or explore the analysis in Jupyter notebook:
```bash
jupyter notebook decisiontree.ipynb
```
=======
# Repo-2_A11.2023.15323
>>>>>>> 06a8299cb6b2989a4fecba3cbecb73a5d2676def
