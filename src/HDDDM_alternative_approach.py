from math import sqrt
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from src.ProbabilityTypes import Probabilities

class HDDDM(BaseDriftDetector):
    """"
    Implementation of the Hellinger Distance Drift Detection Method as proposed by Ditzler and Polikar.
    Paper: http://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/cidue11.pdf
    """

    def __init__(self, distance, features, target, gamma):
        super().__init__()
        self.prev_dist = 0
        self.lambda_ = 0
        self.batch = 1
        self.sum_eps = 0
        self.sum_eps_sd = 0
        self.dist_diff = 0
        self.epsilon_hat = 0
        self.sigma_hat = 0
        self.beta_hat = 0
        self.gamma = gamma
        self.distance = distance
        self.features = features
        self.target = target
        self.reset()

    def reset(self):
        """
        Resets the change detector parameters.
        """
        super().reset()
        self.sum_eps = abs(self.dist_diff)
        self.sum_eps_sd = (abs(self.dist_diff) - self.epsilon_hat) ** 2

    def hard_reset(self):
        """
        Resets the change detector parameters.
        """
        self.prev_dist = 0
        self.lambda_ = 0
        self.batch = 1
        self.sum_eps = 0
        self.sum_eps_sd = 0
        self.dist_diff = 0
        self.epsilon_hat = 0
        self.sigma_hat = 0
        self.beta_hat = 0
        self.reset()

    def generate_prop_dic(self, window, union_values):
        """
        Generates dictionaries with proportions per feature. This enables distance computation between windows.
        :param window: The current window of interest.
        :param union_values: list containing union of unique values of the two windows
        :return: Dictionary with proportions per feature.
        """
        dic = {}
        df = window.value_counts()
        n = window.shape[0]
        for key in union_values:
            if key in window.unique():
                dic.update({key: df.loc[key] / n})
            else:
                dic.update({key: 0})
        return dic

    def generate_dic(self, window, union_values):
        dic = {}
        n = window.shape[0]
        for key in union_values:
            if key in window.keys().unique():
                dic.update({key: window.loc[key] / n})
            else:
                dic.update({key: 0})
        return dic

    def windows_distance(self, ref_window, current_window, posterior = Probabilities.REGULAR):
        """
        Computes the distance between both windows. This approach conditions the distances based on target class values
        :param ref_window: Reference window
        :param current_window: Current window to compare against the reference window
        :return: total distance between the windows
        """
        actual_dist_lst = []
        for i in ref_window[self.target].unique(): # Condition the distance computation on only the current class value

            actual_dist = 0

            ref_tmp = ref_window[ref_window[self.target] == i] #all x values for this class P(x,y=i)
            curr_tmp = current_window[current_window[self.target] == i]

            if not posterior == Probabilities.POSTERIOR:
                prob_ref = len(ref_tmp) / len(ref_window)  # Compute proportions of the class value in the batch
                prob_curr = len(curr_tmp) / len(current_window)
            else:
                prob_ref = 0
                prob_curr = 0

            #doesn't really work, too slow :(

                # ref_x_vals = ref_tmp.value_counts(self.features) #P(x) array
                # curr_x_vals = curr_tmp.value_counts(self.features)
                #
                # ref_val_list = ref_x_vals.keys().tolist()
                # curr_val_list = curr_x_vals.keys().tolist()
                #
                # prob_ref = len(ref_val_list) / len(ref_tmp)
                # prob_curr = len(curr_val_list) / len(curr_tmp)
                #
                # ref_val_list.extend(curr_val_list)
                #
                # ref_dic = self.generate_dic(ref_x_vals, ref_val_list)
                # current_dic = self.generate_dic(curr_x_vals, ref_val_list)
                #
                # actual_dist += self.distance(ref_dic, current_dic)
            for feature in self.features:
                ref_lst_values = ref_tmp[feature].unique()
                current_lst_values = curr_tmp[feature].unique()
                union_values = list(set(ref_lst_values) | set(current_lst_values))
                ref_dic = self.generate_prop_dic(ref_tmp[feature], union_values)
                current_dic = self.generate_prop_dic(curr_tmp[feature], union_values)

                actual_dist += self.distance(ref_dic, current_dic)
                if posterior == Probabilities.POSTERIOR:
                    prob_ref += len(ref_lst_values) / len(ref_tmp)  # Compute proportions of the class value in the batch
                    prob_curr += len(current_lst_values) / len(curr_tmp)
                elif posterior == Probabilities.REGULAR:
                    # APPROACH 2 - FROM PAPER BY SARNELLE, SANCHEZ ET AL.
                    # http://users.rowan.edu/~polikar/research/publications/ijcnn15.pdf

                    actual_dist /= len(self.features)
                    actual_dist_lst.append(actual_dist)


                # APPROACH 3 - FROM DRIFT MAPS
                # https://link.springer.com/article/10.1007/s10618-018-0554-1
            if posterior == Probabilities.MARGINAL:
                actual_dist = actual_dist * (prob_curr + prob_ref) / 2
                actual_dist_lst.append(actual_dist)

            if posterior == Probabilities.MARGINAL or posterior == Probabilities.POSTERIOR:
                final_dist = sum(actual_dist_lst)
            elif posterior == Probabilities.REGULAR:
                final_dist = max(actual_dist_lst)


        return final_dist

    def update(self, ref_window, current_window, warn_ratio, posterior=Probabilities.REGULAR):
        """
        Updating the drift detector per batch
        :param ref_window: Reference window
        :param current_window: Current window
        :param warn_ratio: The ratio at which possible drift should be warned
        :return: whether or not drift is present in the current batch
        """

        if self.in_concept_change:
            self.reset()

        self.in_concept_change = False
        self.in_warning_zone = False

        actual_dist = self.windows_distance(ref_window, current_window, posterior)

        self.dist_diff = actual_dist - self.prev_dist

        self.epsilon_hat = self.sum_eps / (self.batch - self.lambda_)

        self.sigma_hat = sqrt(self.sum_eps_sd / (self.batch - self.lambda_))

        self.beta_hat = self.epsilon_hat + self.gamma * self.sigma_hat

        if abs(self.dist_diff) > self.beta_hat:
            self.lambda_ = self.batch
            self.in_concept_change = True

        elif abs(self.dist_diff) > self.beta_hat * warn_ratio:
            self.sum_eps += abs(self.dist_diff)
            self.sum_eps_sd += (abs(self.dist_diff) - self.epsilon_hat) ** 2
            self.in_warning_zone = True

        else:
            self.sum_eps += abs(self.dist_diff)
            self.sum_eps_sd += (abs(self.dist_diff) - self.epsilon_hat) ** 2

        self.prev_dist = actual_dist
        self.batch += 1

    def add_element(self, input_value):
        """
        Dummy function, as this is needed when using BaseDriftDetector from sk-multiflow.
        However, this function is not usable for this detector, as it only allows for one input variable.
        We use Update instead of Add_element.
        """
        pass