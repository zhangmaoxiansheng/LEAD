import os
import sys

total_list_name = sys.argv[1]
bad_con_name = sys.argv[2]
with open('{}'.format(total_list_name),'r') as f:
    total_list = f.readlines()
with open('{}'.format(bad_con_name),'r') as f:
    bad_con = f.readlines()
total_list_name = total_list_name.split('.')[0]
for i in total_list:
    if i not in bad_con:
        with open('{}_f.txt'.format(total_list_name),'a') as f:
            f.write(i)