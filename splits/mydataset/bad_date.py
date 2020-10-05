import os
with open('./bad_con.txt','r') as f:
    lines = f.readlines()
bad_date = {}
for line in lines:
    bad_date_now = line.split()[0]
    if bad_date_now not in bad_date.keys():
        bad_date['{}'.format(bad_date_now)] = 1
    else:
        bad_date['{}'.format(bad_date_now)] += 1

liv_cam_path = '/mnt/data1/jianing/liv_cam'
for key in bad_date.keys():
    total_len = len(os.listdir(os.path.join(liv_cam_path,key,'depth')))
    bad_date[key] /= total_len 

bad_date = sorted(bad_date.items(), key=lambda a:a[1],reverse=True)
for i in bad_date:
    with open('bad_date_rate.txt','a') as f:
        f.write('{} {}\r\n'.format(i[0],i[1]))