"""
@Author: Cagla Sozen
@Credit : Thomas Boot
@Date: 21/02/2021
"""
import sklearn.metrics
from tqdm import tqdm
from src.ProbabilityTypes import Probabilities
from src.util import *

def run_hdddm(detector, nr_of_batches_list, data, binned_data, warn_ratio,  model=None, visualize = False, prob_type = Probabilities.REGULAR, save_figures=False, dataset_name = '', threshold = 0.05):
    """

    :param detector:
    :param nr_of_batches_list:
    :param data:
    :param binned_data:
    :param warn_ratio:
    :param model:
    :param visualize:
    :param prob_type:
    :param save_figures:
    :param dataset_name:
    :param threshold:
    :return:
    """
    warning_list = []
    drift_list  = []
    magnitude_list = []

    for nr_of_batches in nr_of_batches_list:
        print('Running experiment with window size: ', len(data)/nr_of_batches)
        Batch = np.array_split(data, nr_of_batches)  # Batching the dataset.

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
        drift_type = [] # Stores drift type according to magnitude

        Drift = np.array_split(binned_data, nr_of_batches)  # Always use the discretized data for drift detection!
        Drift_ref = Drift[0]
        detector.hard_reset()

        for i in tqdm(range(1, nr_of_batches)):
            X_batch = Batch[i].iloc[:, 0:-1]
            y_batch = Batch[i].iloc[:, -1]
            Drift_batch = Drift[i]

            if model is not None:
                y_pred = model.predict(X_batch)
                acc = sklearn.metrics.accuracy_score(y_batch, y_pred)
                accuracy.append(acc)

            detector.update(Drift_ref, Drift_batch, warn_ratio, prob_type=prob_type)
            drift_magnitude = detector.windows_distance(Drift_ref, Drift_batch, prob_type=prob_type)
            magnitude.append(drift_magnitude)

            if detector.detected_warning_zone():
                warning.append(i)

            elif detector.detected_change():
                drift.append(i)
                Drift_ref = Drift_batch  # Reset the batch to be used as reference.
                # print(f'Drift detected in batch {i} with drift magnitude {drift_magnitude}')
                detector.reset()

                drift_type.append('HIGH') if drift_magnitude > threshold else drift_type.append('LOW')

                if model is not None:
                    model = model.fit(X_batch, y_batch)  # Retrain the model

            else:
                Drift_ref = pd.concat([Drift_ref, Drift_batch])  # Extend the reference batch.

        print(f'\nOverview of Detected warnings: {warning}')
        print(f'\nOverview of Detected drifts in batches: {drift}')
        print(f'\nOverview of Distance magnitudes: {magnitude}')
        print(f'\nOverview of Drift Types: {drift_type}')


        if visualize:
            if model is not None:
                drift_fig = visualize_drift(accuracy, drift, warning, nr_of_batches, len(X_train), drift_type)
                if save_figures:
                    fig_name = dataset_name + "_window_size_" + str(nr_of_batches) + "_" + "drift_" + str(prob_type.name)
                    drift_fig.savefig('out/' + fig_name + '.png')

            mag_fig = visualize_magnitude(magnitude)
            if save_figures:
                fig_name = dataset_name + "_window_size_" + str(nr_of_batches) + "_" + "mag_" + str(prob_type.name)
                mag_fig.savefig('out/' + fig_name + '.png')

        warning_list.append(warning)
        drift_list.append(drift)
        magnitude_list.append(magnitude)
    return warning_list, drift_list, magnitude_list, drift_type
