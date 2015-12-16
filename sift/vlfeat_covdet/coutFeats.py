import os

root = "/home/yzhou/IvIndex/data/Images"
filenames = os.listdir(root)
filenames.sort()
cnt = 0
for filename in filenames:
    if filename.startswith("ukbench") and filename.endswith(".hessiansift"):
        siftpath = os.path.join(root, filename)
        l = len(open(siftpath).read().splitlines())
        cnt += l
        print siftpath, l

print "%d total features" % cnt
