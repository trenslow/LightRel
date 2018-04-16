import os
import re


results_dir = 'results/'
sigma = 0.0
results_files = [i for i in os.listdir(results_dir) if 'results' in i]
k = len(results_files)
for file in results_files:
    with open(results_dir + file) as imp:
        for line in imp:
            if 'macro-averaged' in line:
                f1_score = re.findall(r'F1 = (.*?)%', line.strip())[0]
                sigma += float(f1_score)

avg = sigma / k
print('results for ' + str(k) + '-fold validation: F1 = {0:.2f}%'.format(avg))