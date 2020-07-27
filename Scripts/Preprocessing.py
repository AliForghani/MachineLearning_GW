import numpy as np
from sklearn import preprocessing
import pandas as pd

raw_csv_data = pd.read_csv("../Input_data/Data.out",delim_whitespace=True)
pd.options.display.max_columns = None
raw_csv_data=raw_csv_data.drop(["DSP"],axis=1)
unscaled_inputs_all=raw_csv_data.iloc[:,0:6]
targets_all=raw_csv_data.iloc[:,6:]
targets_all=targets_all.values

scaled_inputs = preprocessing.scale(unscaled_inputs_all)

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_all[shuffled_indices]

samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

np.savez('../Results/data_train', inputs=train_inputs, targets=train_targets)
np.savez('../Results/data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('../Results/data_test', inputs=test_inputs, targets=test_targets)




