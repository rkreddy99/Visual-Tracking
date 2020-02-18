import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal

scaling_factor = 1 

#Implement this code to detect change from frame to frame

K = 3 ## Number of Guassian normal distributions
threshold = 0.6 ## Threshold value for setting background
alpha = 0.01 ## Learning Rate
height = int(scaling_factor*320) ## Height of input video
width = int(scaling_factor*640) ## Width of input video
num_of_frames = 71 ## There are 71 frames in my video

weights=np.zeros((height,width,K)) ## Probability for observing each pixel
means=np.zeros((height,width, K,3)) ## Mean for each pixel in three colors RGB and three distributions
covar=np.zeros((height,width,K)) ## covar is same for all colors and they're independent
for i in range(height):
    for j in range(width):
        means[i,j]=np.array([[122, 122, 122]]*K) ## updating mean to 122
        covar[i,j]=[36.0]*K 
        weights[i,j]=[1.0/K]*K ##Alloting the weights such that probability is equal for all and their sum is ONE


## Taking input from video
cap=cv2.VideoCapture('test.mp4')
cap.set(3,width)
cap.set(4,height)
## For writing video output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('detector.avi',fourcc, 1.0, (width,height))
N=0 ## Number of frames
while(N<=num_of_frames):
    ret,frame=cap.read()
    N+=1
    if N%7==0:
        frame = cv2.resize(frame, (0,0), fx = scaling_factor, fy = scaling_factor)
        print("number of frames: ", N)
        bkgrnd=np.zeros((height,width),dtype=int) ## For setting background
        for i in range(height):
            for j in range(width):
                bkgrnd[i,j]=-1
                amp=[]
                for k in range(K):
                    amp.append(weights[i,j,k]/np.sqrt(covar[i,j,k])) ## Storing peak amplitude of distrbution
                ord_ind=np.array(np.argsort(amp)[::-1]) ## Sorting in the order of peak amplitude of distributions
                means[i,j]=means[i,j][ord_ind] ## Shuffling in order of amp
                covar[i,j]=covar[i,j][ord_ind]
                weights[i,j]=weights[i,j][ord_ind]
                Fprob=0 ## Cummulative Probability
                for l in range(K):
                    Fprob+=weights[i,j,l]
                    if Fprob>=threshold and l<K-1:
                        bkgrnd[i,j]=l
                        break
                if bkgrnd[i,j]==-1: ## When there is no background pivot
                    bkgrnd[i,j]=K-2 ## Foreground is made as pivot
        out=np.zeros((height,width)) ## initializing the output
        for i in range(height):
            for j in range(width):
                Xt=frame[i,j] ## Sample At time t
                match=-1
                for k in range(K):
                    inv_cov=np.linalg.inv(covar[i,j,k]*np.eye(3)) ## Inverse covariance matrix
                    Xmeans=Xt-means[i,j,k]
                    difference=np.dot(Xmeans.T, np.dot(inv_cov, Xmeans))
                    if difference<6.25*covar[i,j,k]:
                        match=k
                        break

                if match!=-1:  
                    weights[i,j]=(1.0-alpha)*weights[i,j]
                    weights[i,j,match]+=alpha
                    p=alpha * multivariate_normal.pdf(Xt,means[i,j,match],np.linalg.inv(inv_cov))
                    covar[match]=(1.0-p)*covar[i,j,match]+p*np.dot((Xt-means[i,j,match]).T, (Xt-means[i,j,match]))
                    means[i,j,match]=(1.0-p)*means[i,j,match]+p*Xt
                    if match>bkgrnd[i,j]:
                        out[i,j]=250

                else:
                    means[i,j,-1]=Xt
                    out[i,j]=250

        cv2.imshow('FGBG',out)
        cv2.imwrite( "res{}.png".format(N), out );
        out = cv2.merge([out,out,out])
        out = out.astype('uint8')
        video.write(out)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

plt.show()
cap.release()
cv2.destroyAllWindows()