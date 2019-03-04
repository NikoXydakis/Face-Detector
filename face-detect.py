#!/usr/bin/env python

"This program detects if a face is feature in an image that \
is provided in the terminal."
from __future__ import division
import sys, cv2, numpy as np, pylab

#----------------
# Functions     |
#----------------


#------------------------
# find-cloud.py - START |
#------------------------


"At first we want the user to input a source for the images, \
then we need to check if the user input is correct for our \
program to work"

if len (sys.argv) < 2:
    print >>sys.stderr, "Usage:", sys.argv[0], "<image>..."
    sys.exit (1)

"We take the filename of the image that was supplied \
in the terminal and we convert it into a string"
    
fn = sys.argv[1:]
fileName = " ".join(map(str, fn))
#print(fileName) ----> used for debugging purposes
im = cv2.imread (fileName)

isFace = False

# HAAR Cascades

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

eyeGlasses_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye_tree_eyeglasses.xml')

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
face = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in face:
    cv2.rectangle(im, (x,y), (x+w, y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = im[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    isFace = True
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex+ey, ew+eh), (0, 255, 0), 2)
        isFace = True


if(isFace):
    print("yes")
else:
    print("no")
############################################################################

#Image display ----- not needed

"In case the image exceeds 800px of width size \
we resize it and we display the image on the screen."
max_display = 800

ny, nx, nc = im.shape
    
if ny > max_display or nx > max_display:
        nmax = max (ny, nx)
        fac = max_display / nmax
        nny = round_num (ny * fac)
        nnx = round_num (nx * fac)
        im = cv2.resize (im, (nnx, nny)) 
        

cv2.imshow(fileName, im)
cv2.waitKey(2000)
cv2.destroyWindow(fileName)


#-------------------------
# face-detect.py - FINISH |
#-------------------------