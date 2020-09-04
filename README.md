# GenInSAR: Generative CNN-based InSAR Phase Filter
First ever CNN-based generative model for joint phase filtering and coherence estimation, that directly learns the InSAR data distribution. GenInSAR's unsupervised training on satellite and simulated noisy InSAR images outperforms other five related methods in total residue reduction (over 16.5% better on average) with less over-smoothing/artefacts around branch cuts. GenInSAR's Phase, and Coherence Root-Mean-Squared-Error and Phase Cosine Error have average improvements of 0.54, 0.07, and 0.05 respectively compared to the related methods.

Please cite the below [paper](https://doi.org/10.1109/LGRS.2020.3010504) if you use the code in its original or modified form:

*S. Mukherjee, A. Zimmer, X. Sun, P. Ghuman and I. Cheng, "An Unsupervised Generative Neural Approach for InSAR Phase Filtering and Coherence Estimation," IEEE Geoscience and Remote Sensing Letters.*

## Guidelines

Clone this repo and execute the following (Linux) commands in order from your local clone folder:
1. mkdir gsarvenv
2. python3 -m venv ./gsarvenv
3. source gsarvenv/bin/activate
4. pip install -r requirements.txt
5. python geninsar.py

Please remember to set your training dataset folders in [geninsar.py](https://github.com/subhayanmukherjee/geninsar/blob/master/geninsar.py) before running that script, and to rewrite the functions for reading and writing interferograms and coherence map files (in [data_utils.py](https://github.com/subhayanmukherjee/geninsar/blob/master/data_utils.py)) to parse your file formats.

Try out our [InSAR Simulator](https://github.com/Lucklyric/InSAR-Simulator) to train and test GenInSAR on simulated interferograms.
