"""
@Author: Thomas Boot
@Author: Cagla Sozen
@Date: 21/02/2022
"""

from math import sqrt, floor
from sklearn import ensemble
from src.Distance import Distance
from src.HDDDM_alternative_approach import HDDDM
from src.HDDDM_run import run_hdddm, load_dataset, discretize_data
from src.ProbabilityTypes import Probabilities

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #USER DEFINED PARAMETERS

    nr_of_bins = floor(sqrt(1000))  # The number of bins for discretization. Equals the square root of the batch cardinality.
    gamma = 1.5  # Used to define if drift is considered "large enough". See paper for more information.
    warn_ratio = 0.8  # Defines at what point possible drift should be warned.
    categorical_variables = []  # Defines which variables are categorical. DO NOT include target variable!
    to_keep = "all"  # Variables to include in drift computation. Use statement below to specify variables.
    # to_keep = categorical_variables
    prob_type = Probabilities.REGULAR #Approach to be used
    model = ensemble.GradientBoostingClassifier()  # Learner to be used during training
    distance = Distance().hellinger_dist  # Distance metric to be used during drift detection
    nr_of_batches = [1000, 500, 250, 100, 50] #The amount of batches to divide the data in.
    HIGH_DRIFT_THRESHOLD = 0.05 #Threshold used for High Drift Classification

    path = 'data/SEA_Abrupt_high.csv' #Path to the data
    target = 'y'                        #Target variable name

    #END USER DEFINED PARAMETERS

    data, dataset_name = load_dataset(path) #Load dataset. Also returns dataset name as a utility
    binned_data, numerical_cols = discretize_data(data, categorical_variables, nr_of_bins) #Discretize the data as a prerequisite


    if to_keep == "all":
        to_keep = categorical_variables + numerical_cols

    detector = HDDDM(distance, to_keep, target, gamma) #HDDM detector to be used

    warning, drift, magnitude, drift_type = run_hdddm(detector, nr_of_batches, data, binned_data, warn_ratio, model=model,
                                          visualize=True, prob_type=prob_type, save_figures=True,
                                          dataset_name=dataset_name, threshold=HIGH_DRIFT_THRESHOLD) #Run the analysis procedure and return the results
