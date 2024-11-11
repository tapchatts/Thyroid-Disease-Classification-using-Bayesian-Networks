# Thyroid-Disease-Classification-using-Bayesian-Networks
This is the development code used for my honours research report. A Bayesian Approach for Thyroid Disease Prediction: Identifying Key Feature Categories and Symptom Profiles

## File Descriptions

### Data
- `thyroid_data.data`: The dataset file, originally sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/102/thyroid+disease).
- `thyroid_names.names`: A file containing the names and descriptions of all columns present in the data file.

### Model Development
- `Thyroid_Disease_Prediction_Research_Code.ipynb`: Jupyter notebook containing the complete code for model development, training, and evaluation as described in the research.

### Saved Models
- `BDeu_best_model.pkl`: The saved Bayesian Network model of the best-performing BN referenced in the research.
- `random_best_model.pkl`: The saved random BN model after hyperparameter learning, used as a baseline.
- `nn_best_model.pkl`: The saved Neural Network model with masking, also used as a baseline for comparison after hyperparameter learning.

## Loading Saved Models in Python

To reload and use the saved models in Python, use the following code:

```python
import pickle
from pgmpy.models import BayesianNetwork

# Load BDeu best model
BDeu_best_model = BayesianNetwork()
with open('BDeu_best_model.pkl', 'rb') as f:
    BDeu_best_model = pickle.load(f)

# Load random best model
random_best_model = BayesianNetwork()
with open('random_best_model.pkl', 'rb') as f:
    random_best_model = pickle.load(f)

# Load neural network best model
with open('nn_best_model.pkl', 'rb') as f:
    nn_best_model = pickle.load(f)
