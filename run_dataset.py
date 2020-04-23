#pip3 install opencv-python

import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import math

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")#Here we are loading our trained model

img = cv2.imread('steering_wheel_image.jpg',0) #loading our steering wheel image
rows,cols = img.shape

smoothed_angle = 0


#read data.txt
xs = []
ys = []
with open("driving_dataset/data.txt") as f: #Here we are loading our data.txt to get y_i for our test data
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
        ys.append(float(line.split()[1]) * scipy.pi / 180)

#get number of images
num_images = len(xs)


i = math.ceil(num_images*0.8) #Here we are sayig that we want to start from 80th percentile of our data.
print("Starting frameofvideo:" +str(i))

while(cv2.waitKey(10) != ord('q')): #This line says keep running this while loop with 10 millisecond delay until we press key 'q'
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB") #Here we are taking idividual images of the road
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi #Here we are evaluating the image using model to get steering wheel angle
    #call("clear")
    #print("Predicted Steering angle: " + str(degrees))
    print("Steering angle: " + str(degrees) + " (pred)\t" + str(ys[i]*180/scipy.pi) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)) #Here we are showig the individual images of the road
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle) #Here we are smoothing the angle
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1) #This will take my steering image and take the degrees by which I want to rotate this image and put it in the window
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)#This will show my steering wheel
    i += 1

cv2.destroyAllWindows()
