# IMPORTS
from math import sqrt, floor
from sklearn import ensemble
from Distance import Distance
from HDDDM_alternative_approach import HDDDM
from Discretize import Discretizer
from HDDDM_run import run_hdddm
import pandas as pd



# USER DEFINED SETTINGS

#nr_of_batches = 500           # The amount of batches to divide the data in.
nr_bins = floor(sqrt(1000))   # The number of bins for discretization. Equals the square root of the batch cardinality.
gamma = 1.5                   # Used to define if drift is considered "large enough". See paper for more information.
warn_ratio = 0.8              # Defines at what point possible drift should be warned.
categorical_variables = []   # Defines which variables are categorical. DO NOT include target variable!
to_keep = "all"               # Variables to include in drift computation. Use statement below to specify variables.
# to_keep = categorical_variables
POSTERIOR = True

model = ensemble.GradientBoostingClassifier()   # Learner to be used during training
distance = Distance().hellinger_dist   # Distance metric to be used during drift detection

# LOADING THE DATASET

dataset = '../data/SEA_Abrupt_high.csv'
data = pd.read_csv(dataset, delimiter=',', index_col=0)
target = 'y'    # Identify the target class
dataset_name = dataset[dataset.rfind('/')+1:dataset.rfind('.')]
# PREPROCESSING THE DATA

data2 = data.copy()
discretize = Discretizer("equalquantile")  # Choose either "equalquantile" or "equalsize"
discretize.fit(data2, None, to_ignore= categorical_variables)   # Determine which variables need discretization.
numerical_cols = discretize.numerical_cols
binned_data, bins_output = discretize.transform(data2, nr_bins)  # Bin numerical data.

if to_keep == "all":
    detector = HDDDM(distance, categorical_variables + numerical_cols, target, gamma)
else:
    detector = HDDDM(distance, to_keep, target, gamma)


#batch_sizes = [1000, 500, 250, 100, 50]

nr_of_batches = [500]
original_batch_size = nr_of_batches[0]

warning, drift, magnitude, drift_type = run_hdddm(detector, nr_of_batches, data, binned_data, warn_ratio, model=model, visualize=True, posterior=POSTERIOR, save_figures= True, dataset_name=dataset_name)

# # GRADUAL DRIFT TEST

# # Reset the data to be processed to look into smaller parts of the data more in detail.
#
# pair_batch_sizes, pairs = get_drift_pairs(drift, original_batch_size)
#
# Batch = np.array_split(data, original_batch_size)  # Batching the dataset.
# new_data = prepare_data(Batch, pairs[0])
# new_data_2 = new_data.copy()
# new_binned_data, bins_output = discretize.transform(new_data_2, nr_bins)  # Bin numerical data.
#
# grad_model = ensemble.GradientBoostingClassifier()   # Learner to be used during training
# grad_detector = HDDDM(distance, categorical_variables + numerical_cols, target, gamma)
#
# batch_sizes = [1000]
#
# warning, drift, magnitude = run_hdddm(grad_detector, batch_sizes, new_data, new_binned_data, warn_ratio, visualize=True, posterior=POSTERIOR)
