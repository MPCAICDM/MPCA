import os

dir_path = '/data/ljh/anodetec_results/mnist'

tag = 'lsa'

for filename in os.listdir(dir_path):
    if tag in filename:
    p = os.path.join(dir_path, filename)

