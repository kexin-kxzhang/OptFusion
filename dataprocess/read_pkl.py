import pickle
from pathlib import Path

stats_path = Path("./avazu_new/threshold_2/stats")

with open(stats_path.joinpath("feat_map.pkl"), 'rb') as fi:
    feat_map = pickle.load(fi)
with open( stats_path.joinpath("defaults.pkl"), 'rb') as fi:
    defaults = pickle.load(fi)
with open( stats_path.joinpath("offset.pkl"), 'rb') as fi:
    field_offset = pickle.load(fi)
with open( stats_path.joinpath("save_cnt.pkl"), 'rb') as fi:
    cnt = pickle.load(fi)

off = [0]
for i, key in field_offset.items():
    off.append(key)
field_id = []
for i in range(len(off)-1):
    field_id.append([off[i],off[i+1]-1])
print(field_id)

sum = 0
c = 0
li = []
id = 0
id_list = []
for i, key in defaults.items():
    if key < 20000:
        sum = sum+key
        c += 1
    else:
        li.append(i)
        id_list.append(id)
    id += 1

# print(feat_map)
print(cnt['device_id'])
print(defaults)
print(field_offset)
print(cnt)



