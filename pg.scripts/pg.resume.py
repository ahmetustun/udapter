import os
import sys

if len(sys.argv) < 4:
    print("please specify model dir and for each command: time(hours), memory(gb), #cpus")
    exit(1)

'''
module load Python/3.6.4-intel-2018a
module load Anaconda3
. /software/software/Anaconda3/5.3.0/etc/profile.d/conda.sh
conda deactivate
#echo $CONDA_DEFAULT_ENV
conda activate allennlp
#echo $CONDA_DEFAULT_ENV
python train.py --resume logs/udify.baseline.3/2019.10.08_11.05.27
'''


def createJob(name, cmd):
    print("creating: " + name)
    outFile = open(name, 'w')
    outFile.write("#!/bin/bash\n")
    outFile.write('\n')
    outFile.write("#SBATCH --time=" + sys.argv[2] + ":00:00\n")
    outFile.write("#SBATCH --nodes=1\n")
    outFile.write("#SBATCH --ntasks=1\n")
    outFile.write("#SBATCH --mem=" + sys.argv[3] + 'G\n')
    outFile.write("#SBATCH --cpus-per-task=" + sys.argv[4] + '\n')
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
    outFile.write(cmd + "\n")
    outFile.close()


model = sys.argv[1]
name = os.path.split(os.path.split(model)[0])[1] + '.resume'
cmd = 'python train.py --resume ' + model
createJob(name, cmd)
