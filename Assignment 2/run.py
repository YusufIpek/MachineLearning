from subprocess import call
import os

for epoch in(25,50,75):
    for batch_size in (128,256):
        outputfile = os.getcwd() + "\epoch="+str(epoch)+"#batch_size="+str(batch_size)+".txt"
        file = open(outputfile, "w")
        call(["python", os.getcwd() + "\cifar10_cnn.py", str(batch_size), str(epoch)],stdout=file)
        file.close()