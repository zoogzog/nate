import nate as nt

img = "D:/ANDREY/DevNate/test/dbimg/"
file = "D:/ANDREY\DevNate/test\dataset/dataset-seg.txt"
transform = nt.TransformFactory.getTransformSequence(["resize"], [256], True)
transformx = nt.TransformFactory.getTransformSequenceMap(["resize"], [256])


seg = nt.DatagenSegmentation(img, file, transform, transformx)
intensor, outtensor = seg.__getitem__(0)

img = nt.TransformFactory.tensor2image(intensor, True)

img.save("D:/nate/test-yy.png")
#outtensor.save("D:/nate/test-xx.png")

outnmp = outtensor.numpy()
imgo = seg.map2img(outnmp)


imgo.save("D:/nate/test-pp.png")
