import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import statistics


#data = pd.read_csv('test_tester_outputs/X_test_results.csv')
data = pd.read_csv('test_tester_outputs_quant/quantized_model.csv')

d_column_data = data['trace_category']
d_predicted_data = data['number_of_detections']
d_predicted_data = d_predicted_data.astype(int)
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# Iterate through the data
for actual, predicted in zip(d_column_data, d_predicted_data):
    if actual == 'noise' and predicted == 0:
        true_negative += 1
    elif actual == 'earthquake_local' and predicted >= 1:
        true_positive += 1
    elif actual == 'earthquake_local' and predicted == 0:
        false_negative += 1
    elif actual == 'noise' and predicted >= 1:
        false_positive += 1

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)

print("True Positive:", true_positive)
print("True Negative:", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

print("#########################")
##############################################################


p_actual = data['p_arrival_sample']
p_predicted = data['P_pick']
p_error = data['P_error']


true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
p_error_array = np.array([])


for i in range(len(d_column_data)):
    if abs(p_error[i]) <= 50:
        true_positive += 1
        p_error_array = np.append(p_error_array, p_error[i])
    elif(np.isnan(p_actual[i]) and np.isnan(p_predicted[i])):  
        true_negative += 1
    elif(np.logical_not(np.isnan(p_actual[i])) and np.isnan(p_predicted[i])):
        false_negative += 1
    elif(abs(p_error[i]) > 50 or np.logical_not(np.isnan(p_predicted[i]))):
        false_positive += 1


precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)

print("True Positive:", true_positive)
print("False Positive:", false_positive)

print("True Negative:", true_negative)
print("False Negative:", false_negative)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

mean_error= np.nanmean(p_error)
print("P Mean Error:", mean_error/100)
std_error = np.nanstd(p_error)
print("P Standard Deviation Error:", std_error/100)
mae = np.mean(np.abs(p_error_array)/100)
print("Mean Absolute Error (MAE):", mae)


print("#########################")

###########################################################


s_actual = data['s_arrival_sample']
s_predicted = data['S_pick']
s_error = data['S_error']



true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
s_error_array = np.array([])

for i in range(len(d_column_data)):
    if abs(s_error[i]) <= 50:
        true_positive += 1
        s_error_array = np.append(s_error_array, s_error[i])

    elif(np.isnan(s_actual[i]) and np.isnan(s_predicted[i])):  
        true_negative += 1
    elif(np.logical_not(np.isnan(s_actual[i])) and np.isnan(s_predicted[i])):
        false_negative += 1
    elif(abs(s_error[i]) > 50 or np.logical_not(np.isnan(s_predicted[i]))):
        false_positive += 1


precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1 = 2 * (precision * recall) / (precision + recall)

print("True Positive:", true_positive)
print("True Negative:", true_negative)
print("False Positive:", false_positive)
print("False Negative:", false_negative)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


mean_error= np.nanmean(s_error)
print("S Mean Error:", mean_error/100)
std_error = np.nanstd(s_error)
print("S Standard Deviation Error:", std_error/100)
mae = np.mean(np.abs(s_error_array)/100)
print("Mean Absolute Error (MAE):", mae)
