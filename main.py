import nltk
nltk.download("popular") # importing dataset
from nltk.corpus import twitter_samples
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sentiment_analysis as sentiment_analysis

positive_sample = twitter_samples.strings('positive_tweets.json')
negative_sample = twitter_samples.strings('negative_tweets.json')

# How many samples do we have ?
print(len(positive_sample)) # 5000
print(len(negative_sample)) # 5000


# 1) Training and Testing Arrays setup 
train_x = positive_sample[:4000] + negative_sample[:4000]
test_x = positive_sample[4000:] + negative_sample[4000:]

# Combine positive and negative labels
train_y = np.append(np.ones((len(positive_sample[:4000]), 1)), np.zeros((len(negative_sample[:4000]), 1)), axis=0)
test_y = np.append(np.ones((len(positive_sample[4000:]), 1)), np.zeros((len(negative_sample[4000:]), 1)), axis=0)


# 2) Train model using the training dataset and test it on the test dataset
freqs = sentiment_analysis.build_freqs(positive_sample, negative_sample)
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = sentiment_analysis.features_extraction(train_x[i], freqs)
print(X)
"""
[[1.000e+00 3.764e+03 7.200e+01]
 [1.000e+00 4.464e+03 5.170e+02]
 [1.000e+00 3.759e+03 1.600e+02]
 ...
 [1.000e+00 1.840e+02 9.890e+02]
 [1.000e+00 2.560e+02 4.855e+03]
 [1.000e+00 2.400e+02 4.967e+03]]
"""

theta = sentiment_analysis.learn(X, train_y, 1e-9, 1000)
print(theta)
"""
[[ 4.80560248e-08]
 [ 4.29616097e-04]
 [-4.54043371e-04]]
"""


# 3) Compute accuracy 
# - y_hat    : y predicted
# - accuracy : accuracy value [0,1]
y_hat, accuracy = sentiment_analysis.compute_accuracy(test_x, test_y, freqs, theta)
print(f"Accuracy : {accuracy}") # Accuracy : 0.997


# 4) Other custom tests
for str in ["Today is beautiful day!! :)",          # prediction = 1 (positive)
            "I am hopeless for the next week :(",   # prediction = 0 (negative)
            "That decision was a shame!"]:          # prediction = 0 (negative)
    print(f"{str} - {sentiment_analysis.predict(str, freqs, theta)}")


# 5) Confusion matrix plot
data = {
    'y_actual'    : test_y.reshape(len(y_hat)),
    'y_predicted' : y_hat
}
df = pd.DataFrame(data, columns=['y_actual', 'y_predicted'])
confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
plt.show()