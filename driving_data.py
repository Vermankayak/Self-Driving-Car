import scipy.misc
import random

xs = []
ys = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0 #in this code test is ofte referred to as validation

#read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0]) #In xs we are appeding the path of each of the image files. 
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)#In ys I am appending the angles in radians

#get number of images
num_images = len(xs)

#Below I am creating my training and validation data using temporal split or time based splitting
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

#Computing the number of training and test images.
num_train_images = len(train_xs)
num_val_images = len(val_xs)

#The below function loads one batch of images from the training data
def LoadTrainBatch(batch_size):
    global train_batch_pointer #After lets say we get a batch size of 100 images in train dataset in 1st iteration the train_batch_pointer will point to the location of 100th image so that the next batch can be taken from 101th image and so on.
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(train_xs[(train_batch_ pointer + i) % num_train_images])[-150:], [66, 200]) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out
    
#The below function loads one batch of images from the validation or test data
def LoadValBatch(batch_size):
    global val_batch_pointer #After lets say we get a batch size of 100 images in test dataset in 1st iteration the test_batch_pointer will point to the location of 100th image so that the next batch can be taken from 101th image and so on.
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(scipy.misc.imresize(scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], [66, 200]) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
