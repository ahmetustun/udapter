import json
import os
import sys

if len(sys.argv) < 2:
    print("please specify log dir")
    exit(1)


las_results = dict()
uas_results = dict()

prefix = 'test.conllu_'
suffix = '.json'
log_dir = sys.argv[1]

for f in os.listdir(log_dir):
    if f.startswith(prefix) and f.endswith(suffix):
        lang = f.split(prefix)[1].split(suffix)[0]
        with open(os.path.join(log_dir,f)) as j:
            data = json.load(j)
            las = data['LAS']['aligned_accuracy']
            uas = data['UAS']['aligned_accuracy']
            las_results[lang] = float(las)*100
            uas_results[lang] = float(uas)*100

las_results = {k: v for k, v in sorted(las_results.items(), key=lambda item: item[0])}
uas_results = {k: v for k, v in sorted(uas_results.items(), key=lambda item: item[0])}
hr_langs = ['ar', 'eu', 'zh', 'en', 'fi', 'he', 'hi', 'it', 'ja', 'ko', 'ru', 'sv', 'tr']

hr_out = open(os.path.join(log_dir, 'hr_results.txt'), 'w')
lr_out = open(os.path.join(log_dir, 'lr_results.txt'), 'w')
for l,r in las_results.items():
    if l in hr_langs:
        hr_out.write('{}, {:.2f}, {:.2f}\n'.format(l,r,uas_results[l]))
    else:
        lr_out.write('{}, {:.2f}, {:.2f}\n'.format(l,r,uas_results[l]))

hr_out.close()
lr_out.close()
