# 1BM130-Group2
This repository contains the data and code of group 2 of the course Design of Data-Driven Business Operations at Eindhoven University of Technology (TU/e) 2020/2021
## Running the code
To run the code, you can simply run the py files. To inspect the outputs, in editors such as VSCode and Pycharm, you can run the code sections between `# %%`, like cells of a Jupyter Notebook.
### Data
All data files, both the initial files and all intermediately created files, are present, so each file can be run in any order. Besides, some pickled variables are stored here. Note, running some files may take quite long.

### Models
The models folder contains the decision tree models and the columns that were used for these models. These are used in `lp.py`
### Preprocessing
In `merge_data.py` an initial data preparation is performed, providing the basis for the remainder of the analysis. `constants.py` contains some constant dictionaries and lists that are used in this and several other files.

### Descriptive
The descriptives are created in `descriptive.py`

### Behavioural
Both preprocessing and clustering for the behavioural analysis are performed in `behavioural.py`

### Predictive
The preprocessing for the predictive (and prescriptive) modeling is performed in `predictive_preprocessing.py`. Then, the experiments/model development are performed in `predictive_modeling.py`. `helpers.py` contains a function for running hyperopt, which is used in `predictive_modeling.py`
### Prescriptive
For the prescriptive analytics, two files are used. `lpclasslibrary.py` contains classes that are used in `lp.py`. You can run `lp.py` to run an optimization for an auction. At the top of this file, the id of the auction can be entered (which must be one of the auctions in the sample data) 

### Required packages
- gurobipy >= 9.1.2
- hyperopt >= 0.2.5
- matplotlib >= 3.3.2
- numpy >= 1.19.5
- pandas >= 1.1.3
- scipy >= 1.5.4
- seaborn >= 0.11.1
- sklearn >= 0.0
- tqdm >= 4.56.2

