import os
import numpy as np
import pdb
import sys
bad_seq_n = sys.argv[1]
bad_con_n = sys.argv[2]
origin_list = sys.argv[3]
output_list = sys.argv[4]

with open(bad_seq_n,'r') as f:
    lines = f.readlines()
indexes = []
losses = []
for line in lines:
    index,loss = line.split()
    loss = float(loss)
    index = index.replace('[','')
    index = index.replace(']','')
    index = int(index)
    
    indexes.append(index)
    losses.append(loss)

losses = np.asarray(losses)
average_loss = np.mean(losses)

print(average_loss)
bad_seq = losses > average_loss
#pdb.set_trace()
count = 0
with open(origin_list,'r') as f:
    con = f.readlines()
for ind,b in enumerate(bad_seq):
    if b:
        ii = indexes[ind]
        with open(bad_con_n,'a') as f:
            f.write('{}'.format(con[ii]))
    else:
        count += 1
        ii = indexes[ind]
        with open(output_list,'a') as f:
            f.write('{}'.format(con[ii]))
print(count)

