import sys

fold = sys.argv[1]
sample_rate = sys.argv[2]

fp = open('setting_file', 'w')
fp.write('fold '+fold+'\n')
fp.write('sample_rate '+sample_rate)
