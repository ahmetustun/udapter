import os
import sys

if len(sys.argv) < 5:
    print("please specify model file, test dir and for each command: time(hours), memory(gb), #cpus")
    exit(1)


def createJob(name, cmd):
    print("creating: " + name)
    outFile = open(name, 'w')
    outFile.write("#!/bin/bash\n")
    outFile.write('\n')
    outFile.write("#SBATCH --time=" + sys.argv[3] + ":00:00\n")
    outFile.write("#SBATCH --nodes=1\n")
    outFile.write("#SBATCH --ntasks=1\n")
    outFile.write("#SBATCH --mem=" + sys.argv[4] + 'G\n')
    outFile.write("#SBATCH --cpus-per-task=" + sys.argv[5] + '\n')
    outFile.write("#SBATCH --job-name=" + name + '\n')
    outFile.write("#SBATCH --output=" + name + '.log\n')
    outFile.write("#SBATCH --partition=gpu\n")
    outFile.write("#SBATCH --gres=gpu:v100:1\n")
    outFile.write("\n")
    outFile.write("module load Python/3.6.4-intel-2018a\n")
    outFile.write("module load Anaconda3\n")
    outFile.write(". /software/software/Anaconda3/5.3.0/etc/profile.d/conda.sh\n")
    outFile.write("conda deactivate\n")
    outFile.write("conda activate allennlp\n")
    outFile.write("conda activate allennlp\n")
    for c in cmd:
        outFile.write(c + "\n")
    outFile.close()


model = sys.argv[1]
test_dir = sys.argv[2]
cmd = []
ts = os.listdir(test_dir)
for t in ts:
    if 'test' in t:
        test = os.path.join(test_dir, t)
        str = 'python predict.py ' + model + ' ' + test + ' ' + os.path.join(os.path.split(model)[0], t) + ' --eval ' + os.path.join(os.path.split(model)[0], t) + '.json'
        cmd.append(str)

name = model.split('/')[-3] + '.predict'
createJob(name, cmd)
