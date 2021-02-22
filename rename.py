import os

path = 'background/'

f = os.listdir(path)

n = 0

for i in f:

    old_name = path + f[n]

    new_name = path + str(n+1).zfill(2) + '.jpg'

    os.rename(old_name, new_name)
    print(old_name, '======>', new_name)

    n += 1
