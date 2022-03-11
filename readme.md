# READ ME

In context of online learning data comes in streams and hereby, one of the biggest challenges that a model faces is to be able preserve its optimality. Real-world data often comes from non-stationary environments and evolves over time, leading to a change in its distribution. This phenomenon is called concept drift. In certain cases, concept drift might be particularly disruptive to the optimality of Machine Learning systems. Once the model becomes sub-optimal, it needs to be re-configured. For an improved re-configuration process, understanding and quantifying drift is essential. Hereby, this research is focused on analysing and quantifying drift using Hellinger Distance Based Drift Detection. Extending previous work of a fellow student Thomas Boot, class drift is analyzed using posterior class probabilities and the effect of window size to the accuracy of drift detection and measurement is explored. According to the experiment results, an abrupt drift classification method is proposed according to the magnitude of the detected drift. Code developed for this project can be used as a library for drift analysis and classification with small improvements.

## Requirements

This repository uses Python 3.9. Dependencies required by this project can be installed via

`pip install -r requirements.txt`

## Usage 

- main.py is the main function used during development. Parameters can be changed from the **user-defined variables** block inside the main.py script.
- Extracting the /data folder from Cagla_Sozen_data.zip is sufficient for setting the default data path.
- Plots generated will be outputted in the /out folder. To change the location, HDDDM_run (function _run_hdddm_) script should be changed. 
- Custom datasets with drift can be generated using the GenerateDatasets script. Currently it contains the generation code for all the datasets used for the experiments. 

## Details
- /src contains the core classes and scripts containing the functions required for the project. 
  - Discretize.py contains a class for instantiating a Discretizer with the selected method.
  - Distance.py contains the distance functions, Hellinger Distance, Total Variation Distance, KL_divergence, Jensen Shannon Divergence.
  - HDDDM_alternative_approach.py contains an implementation of the original HDDM algorithm as proposed by Ditzler and Polikar [1] with added different approaches [2], [3]. 
  - HDDDM_run script contains the function to run HDDDM with the selected parameters.
  - ProbabilityTypes.py contains the enumerated types of Approaches (Probabilities) [2], [3].
  - util.py contains utility functions
- /out is the default output folder for the plots generated.
- /data contains the data to be used as .csv files

## References 

[1] G. Ditzler and R. Polikar. Hellinger distance based drift detection for  nonstationary environments. In 2011 IEEE Symposium on Computational Intelligence in Dynamic and Uncertain Environments (CIDUE), pp. 41–48, 2011. doi: 10.1109/CIDUE.2011.5948491

[2] J. Sarnelle, A. Sanchez, R. Capo, J. Haas, and R. Polikar. Quantifying  the limited and gradual concept drift assumption. In 2015 International Joint Conference on Neural Networks, IJCNN 2015, Proceedings of the  International Joint Conference on Neural Networks. Institute of Electrical  and Electronics Engineers Inc., United States, Sept. 2015. International Joint Conference on Neural Networks, IJCNN 2015 ; Conference date: 12-07-2015 Through 17-07-2015. doi: 10.1109/IJCNN.2015.7280850

[3] G. Webb, L. Lee, B. Goethals, and F. Petitjean. Analyzing concept drift  and shift from sample data. Data Mining and Knowledge Discovery, 32, 09 2018. doi: 10.1007/s10618-018-0554-1
