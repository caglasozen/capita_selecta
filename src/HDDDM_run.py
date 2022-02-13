import sklearn.metrics
import pandas as pd
import numpy as np
from math import sqrt, floor
from matplotlib import pyplot as plt
from sklearn import ensemble
from Distance import Distance
from HDDDM_alternative_approach import HDDDM
from Discretize import Discretizer
from util import visualize_drift, visualize_magnitude


def run_hdddm(detector,nr_of_batches_list, Data, binned_data, warn_ratio,  model=None, visualize = False, posterior = False):
    for nr_of_batches in nr_of_batches_list:
        Batch = np.array_split(Data, nr_of_batches)  # Batching the dataset.

        if model is not None:
            # INITIAL MODEL TRAINING

            X_train = Batch[0].iloc[:, 0:-1]
            y_train = Batch[0].iloc[:, -1]

            model.fit(X_train, y_train)

        # ITERATING THROUGH THE BATCHES + DRIFT DETECTION

        drift = []      # Stores detected drifts
        warning = []    # Stores detected warnings prior to drift.
        accuracy = []   # Stores model accuracy per batch.
        magnitude = []  # Stores the magnitude of change between batches.

        Drift = np.array_split(binned_data, nr_of_batches)  # Always use the discretized data for drift detection!
        Drift_ref = Drift[0]

        for i in range(1, nr_of_batches):
            print('Processing batch: ', i)
            X_batch = Batch[i].iloc[:, 0:-1]
            y_batch = Batch[i].iloc[:, -1]
            Drift_batch = Drift[i]

            if model is not None:
                y_pred = model.predict(X_batch)
                acc = sklearn.metrics.accuracy_score(y_batch, y_pred)
                accuracy.append(acc)

            detector.update(Drift_ref, Drift_batch, warn_ratio, posterior=posterior)
            drift_magnitude = detector.windows_distance(Drift_ref, Drift_batch, posterior=posterior)
            magnitude.append(drift_magnitude)

            if detector.detected_warning_zone():
                warning.append(i)

            elif detector.detected_change():
                drift.append(i)
                Drift_ref = Drift_batch  # Reset the batch to be used as reference.
                # print(f'Drift detected in batch {i} with drift magnitude {drift_magnitude}')
                detector.reset()

                if model is not None:
                    model = model.fit(X_batch, y_batch)  # Retrain the model

            else:
                Drift_ref = pd.concat([Drift_ref, Drift_batch])  # Extend the reference batch.

        # print(f'\nOverview of Detected warnings: {warning}')
        print(f'\nOverview of Detected drifts in batches: {drift}')
        print(f'\nOverview of Distance magnitudes: {magnitude}')

        if visualize:
            if model is not None:
                visualize_drift(accuracy, drift, warning, nr_of_batches, len(X_train))
            visualize_magnitude(magnitude)

        return warning, drift, magnitude

