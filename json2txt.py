import json
import os

filedir="/media/zhujun/share/MVS/scene0707-00"
json_file=os.path.join(filedir,"pair.json")
txt_file=json_file.replace('json','txt')
if not os.path.exists(json_file):
    raise IOError(json_file,'not found!!')
with open(json_file,'r') as f:
    pair_data=json.load(f)
with open(txt_file, 'w') as f:
        f.write('%d\n' % len(pair_data))
        for i, ref_view in enumerate(pair_data.keys()):
            # if i+1>20: break
            f.write('%d\n%d ' % (int(ref_view), len(pair_data[ref_view])))
            for image_id in pair_data[ref_view]:
                f.write('%d %d ' % (int(image_id), 100))
            f.write('\n')
# for ref_view in pair_data.keys():
#     self.metas += [(scan,int(ref_view), [int(x) for x in pair_data[ref_view]])]