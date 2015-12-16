import os

root = "/home/yzhou/IvIndex/data/Images"
filenames = os.listdir(root)
filenames.sort()
for filename in filenames:
    if filename.startswith("ukbench") and filename.endswith(".jpg"):
        imgPath = os.path.join(root, filename)
        outPath = os.path.join(root, filename[:filename.rfind(".")] + ".hessiansift")
        cmd = "./sift --estimateaffineshape --estimateorientation --peakthreshold 0.0035 %s %s" % (imgPath, outPath)
        print cmd
