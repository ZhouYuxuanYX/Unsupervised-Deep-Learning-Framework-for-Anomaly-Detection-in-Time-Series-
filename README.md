# Unsupervised-Deep-Learning-Framework-online-and-offline-for-Anomaly-Detection-in-Time-Series-
Unsupervised deep learning framework with both online(MLP: prediction-based, 1 D Conv and VAE: reconstruction-based) and offline(Wavenet: in-sample prediction) settings for anaomaly detection in time series data

## Unspupervised learning

Approach with unsupervised learning: without giving any label for normal or abnormal examples, the anomaly detection problem is formulated in another way: either by re-constructing the given input or predicting unseen examples given only part of the data set, a sequence of errors between the original data set and generated data set could be acquired. Then based on these errors, anomaly scores could be calculated via e.g. mean square errors.

Among these unsupervised methods, two main approaches are to be implemented and investigated, namely prediction-based and
reconstruction-based anomaly detection in times series data:

• Prediction-based method: the models are given a segment of time series data, and the
output shall be the predicted value of next few successive points based on the previous
segment.
• reconstruction-based method: Techniques such as auto-encoding or Principal Compo-
nent Analysis first map the input data to a lower-dimensional space and then try to
reconstruct it again without losing the main information.
