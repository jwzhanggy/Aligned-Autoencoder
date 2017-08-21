import sys
import numpy as np

sample_rate = sys.argv[1]

p = np.array([])
r = np.array([])
f1 = np.array([])
acc = np.array([])
auc = np.array([])

result_file = 'results/sample_result_'+str(sample_rate)
fp = open(result_file, 'r')
for line in fp:
    arr = line.strip().split('\t')
    p = np.append(p, float(arr[0]))
    r = np.append(r, float(arr[1]))
    f1 = np.append(f1, float(arr[2]))
    acc = np.append(acc, float(arr[3]))
    auc = np.append(auc, float(arr[4]))

report_file = 'reports/sample_report_'+str(sample_rate)
fr = open(report_file, 'w')
fr.write(str(np.mean(p))+'\t'+str(np.mean(r))+'\t'+str(np.mean(f1))+'\t'+str(np.mean(acc))+'\t'+str(np.mean(auc))+'\n')
fr.write(str(np.std(p))+'\t'+str(np.std(r))+'\t'+str(np.std(f1))+'\t'+str(np.std(acc))+'\t'+str(np.std(auc))+'\n')
