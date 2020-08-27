import os
import sys

if len(sys.argv) < 2:
    print("please specify log dir")
    exit(1)

model_file = 'model.tar.gz'
model_state = 'model_state_epoch_'
training_state = 'training_state_epoch_'
best_ckpt = 'best.th'
save_dir = 'midpoints'

log_dir = sys.argv[1]

files_and_dirs = os.listdir(log_dir)


def find_last_epoch(log_dir, model_state):
    last = 0
    for f in os.listdir(log_dir):
        if f.startswith(model_state):
            ep = int(f.split('.')[0].split(model_state)[1])
            last = ep if ep > last else last
    return str(last) + '.th'


def clean_dir(log_dir, model_state, training_state):
    for f in os.listdir(log_dir):
        if f.startswith(model_state) or f.startswith(training_state):
            os.remove(os.path.join(log_dir, f))


if model_file in files_and_dirs:
    clean_dir(log_dir, model_state, training_state)
else:
    if save_dir not in files_and_dirs:
        os.makedirs(os.path.join(log_dir, save_dir))
    last = find_last_epoch(log_dir, model_state)
    os.replace(os.path.join(log_dir, model_state+last), os.path.join(log_dir, save_dir, model_state+last))
    os.replace(os.path.join(log_dir, training_state + last), os.path.join(log_dir, save_dir, training_state + last))
    os.replace(os.path.join(log_dir, best_ckpt), os.path.join(log_dir, save_dir, best_ckpt.split('.')[0]+'.'+last))
    clean_dir(log_dir, model_state, training_state)
    os.replace(os.path.join(log_dir, save_dir, model_state + last), os.path.join(log_dir, model_state + last))
    os.replace(os.path.join(log_dir, save_dir, training_state + last), os.path.join(log_dir, training_state + last))