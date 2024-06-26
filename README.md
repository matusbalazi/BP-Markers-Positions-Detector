# Markers Positions Detector
### _for Hangprinter's Computer Vision System_
<p align="justify">
  For those who are currently building Computer Vision System for their Hangprinters. I made a DLC to Tobben's work on auto calibration simulation for Hangprinter.<br><br>
  <b><i>Markers Positions Detector</i></b> is an independent program which detects markers placed on the Hangprinter's effector (or on the "mover") and calculates distances between them.<br><br>
  <b><i>Main goal of this detector is to automate and simplify process of obtaining markers positions.</i></b><br><br>
  As an output is XML file which is generated by program itself and it contains XYZ coordinates of each marker. This XML file is later used by 
  <i><a href=https://gitlab.com/tobben/hpm>hpm</a></i> program, but it's not necessary for using Markers Positions Detector.<br><br><br>
</p>

## Preparations
<p align="justify">
  It is assumed that you have python installed on your computer. If you don't, follow instructions below.
</p>

```
sudo apt update
sudo apt install python3
```

<p align="justify">
  And check the version of your currently installed python.
</p>

```
python3 --version
```

<p align="justify"> 
  If you succeed, you can move on. Clone this repository with:
</p>

```
git clone https://github.com/matusbalazi/BP-Markers-Positions-Detector
```

<p align="justify">
  I recommend using PyCharm IDE, but also ordinary command line will work well.<br><br>
  You will also need to install the following packages: 
  <i><a href="https://github.com/opencv/opencv-python">opencv</a>, 
    <a href="https://github.com/numpy/numpy">numpy</a>, 
    <a href="https://github.com/scipy/scipy">scipy</a>, 
    <a href="https://github.com/uqfoundation/mystic">mystic</a>, 
    <a href="https://github.com/python-pillow/Pillow">pillow</a>, 
    <a href="https://github.com/tartley/colorama">colorama</a>.</i><br><br>
  If you have PyCharm IDE just hit <i>Alt+Enter</i> when your cursor is placed on the red underlined package and select <i>Install package</i>.<br><br>
  If you have only command line interface follow these commands:
</p>

```
sudo apt update

# install cv2 package
sudo apt install python3-opencv

# install numpy package
sudo apt install python3-pip
sudo pip install numpy

# install scipy package
sudo apt install python3-scipy

# install mystic package
sudo pip install mystic

# install PIL package
sudo pip install pillow

# install colorama package
sudo pip install colorama
```

<p align="justify">
  To run Markers Positions Detector program type to Terminal following:
</p>

```
python ./main.py
```

<p align="justify">
  or
</p>

```
python3 ./main.py
```

<p align="justify">
  There is a bunch of arguments you can add to speed up later process:
<ol align="justify">
  <li><i>-i</i> or <i>--image</i> means path to the input image</li>
  <li><i>-wi</i> or <i>--width</i> means width of the output image</li>
  <li><i>-he</i> or <i>--height</i> means height of the output image</li>
  <li><i>-c</i> or <i>--coords</i> means comma separated list of source points to be transformed (write all this list in quotes)</li>
</ol>
  So the command to run Markers Positions Detector program can look like this:
</p>

```
python ./main.py --image input_images_examples/image3.jpg --width 1438 --height 1200 --coords "[(253,572),(870,214),(545,1073),(1162,715)]"
```

<p align="justify">
  You will also need some photo of your markers placed on the effector. You can take a photo with any camera. 
  I'm using same camera as Tobben (Arducam 8MP Sony IMX219 camera module with M2504ZH05 Arducam lens). 
  I recommend to reduce shutter time while taking a photo to avoid unwanted noise in the image.
  But our algorithm should work well with any value of shutter. If you have some kind of PiCam and you have installed correct version of 
  <i><a href="https://www.raspberrypi.com/documentation/accessories/camera.html">raspistill</a></i> which matches your lens you can take a photo by typing:
</p>

```
raspistill --quality 100 --timeout 300 --shutter 100000 --ISO 50 --width 3280 --height 2464
```

<p align="justify">
  Lower value of shutter makes brighter image, higher makes darker image.<br><br>
  The algorithm relies on the photo being taken from the location opposite to anchor B of Hangprinter to detect the markers in the correct order 
  so the user does not have to do anything. Otherwise if the photo is taken from another angle (from location opposite to anchor A or C) 
  you will have to rearrange calculated distances in the text file to correct order.<br><br>
  Markers should be detected in the order as shown in the image below.
</p>

![pozicie_znaciek_no3](https://user-images.githubusercontent.com/91671608/162260252-c1b9677c-21ef-47b1-964a-a3a10a4508d7.png)

<p align="justify">
  Where the individual tag numbers stand for:
<ul>
  <li><i>0</i> - nozzle</li>
  <li><i>1-6</i> - markers numbered from 0 to 5</li>
</ul>
</p>

<p align="justify">
To detect nozzle position I pushed the pin to coupling on the extruder and screwed (or you can glue it) reflective disc to this pin.
In the <i><a href="https://github.com/matusbalazi/markers_positions_detector/tree/master/stl">stl</a></i> folder of this repo you will
find stls of pins. They have different lengths and tolerances. Choose the one that suits you best.<br>
</p>

<p align="justify">
Markers Positions Detector detects markers (probably you have reflective discs, but any circular shape will work) and calculates 21 distances between them in the following order:
</p>

<p align="justify">
<ul>
  <li>0->1, 0->2, 0->3, 0->4, 0->5, 0->6,</li>
  <li>1->2, 1->3, 1->4, 1->5, 1->6,</li>
  <li>2->3, 2->4, 2->5, 2->6,</li>
  <li>3->4, 3->5, 3->6,</li>
  <li>4->5, 4->6,</li>
  <li>5->6.</li>
</ul>
</p>

<p align="justify">
If you have everything ready, let's run the application.<br><br><br>
</p>

## User Guide
<p align="justify">
  I'll show you only some basic steps to use. The application gives you enough help while using it. This is only CLI application.
  In the main menu you will find 4 options to choose from:
  <ol align="justify">
  <li><i>Perspective transformation</i></li>
    <ul>If the angle at which you took the photo is too large it is sometimes beneficial to transform the image to "bird's-eye view".</ul><br>
  <li><i>Circle detector w/distance calculator (OpenCV)</i></li>
    <ul>Detects circles and calculates distances between them using OpenCV functions. This detection is fast but sometimes it isn't so reliable.</ul><br>
  <li><i>Circle detector w/distance calculator</i></li>
    <ul>Detects circles and calculates distances between them, but doesn't use OpenCV functions. This detection is slow but sometimes a little bit more reliable.</ul><br>
  <li><i>End detection and start analyzing data</i></li>
    <ul>After succesful detection analyzes measurements and generates XML file with XYZ coordinates of each marker on the effector.</ul><br>
  </ol>
  For example let's choose second option.
</p>

<p align="center">

<img src="https://user-images.githubusercontent.com/91671608/162267205-c56895b7-f110-4d3a-938a-22650cb3bb9f.png">
  
</p><br>

<p align="justify">
  Program asks if we want to analyze original or transformed image.
</p>

<p align="center">

<img src="https://user-images.githubusercontent.com/91671608/162272162-d100f334-caa4-4e58-8728-f8a3a014496d.png">
  
</p><br>

<p align="justify">
  We need to enter the necessary parameters and choose the detection method. After that detection will start.
</p>

<p align="center">

<img src="https://user-images.githubusercontent.com/91671608/162271966-b76c6a44-9173-4208-8c95-ef8827615e0b.png">
  
</p><br>
  
<p align="justify">
  If no circles detected or the required number of circles was not detected we can run advanced circle detection.
</p>

<p align="center">

<img src="https://user-images.githubusercontent.com/91671608/162272298-1eb5e732-053c-4c5d-bc27-7c2734143390.png">
  
</p><br>
  
<p align="justify">
  All circles were detected, distances between them were calculated and written to text file, output image was saved. Check the order of detected circles. It must match the correct order as was shown in the <i>Preparations</i> part.
Output image is little bit messy but later take a look in the text file <i>result.txt</i>. There are written calculated distances. To continue hit the <i>Enter</i> to close the image.
</p>

<p align="center">
  
<img src="https://user-images.githubusercontent.com/91671608/162270343-780343a3-e446-44fe-8d12-5f00594f2490.png">
  
</p><br>

<p align="justify">
  Now we can start to analyze data that we got to achieve markers positions.
</p>

<p align="center">
  
<img src="https://user-images.githubusercontent.com/91671608/162272750-ed513294-4681-44aa-a301-831fb25aaf56.png">
  
</p><br>

<p align="justify">
  Here begins Tobben's script which I edited. We got best intermediate cost and positions of our markers, but truly important for us is to get best 
  final cost and positions. To get that we have to manually measure Z height (in milimeters), which is distance between nozzle plane and markers plane.
</p>

<p align="center">
  
<img src="https://user-images.githubusercontent.com/91671608/162273859-ef15475d-c2fd-47bb-8619-f3e299e86dcd.png">
  
</p><br>

<p align="justify">
  Finally we got what we wanted all the time. XYZ positions of markers on the effector. Value of best final cost should be less than 3 so that
  we can consider the results are reliable.
</p>

<p align="center">
  
<img src="https://user-images.githubusercontent.com/91671608/162274519-4ee2835e-8284-4035-9e77-658bd2adf161.png">
  
</p><br>

<p align="justify">
  This is how the generated XML file with marker parameters looks like. This file is important if we want to use hpm program.
</p>

<p align="center">
  
<img src="https://user-images.githubusercontent.com/91671608/162275351-7cf7f97f-1862-41c6-a149-32770f1aa819.png">
  
</p>

<br><p align="justify">
  That was a quick look at Markers Positions Detector program. I hope it will help you. You can try another detection methods. 
  They work very similarly. Also look at <i><a href="https://github.com/matusbalazi/markers_positions_detector/tree/master/input_images_examples">input_images_examples</a></i>
  and <i><a href="https://github.com/matusbalazi/markers_positions_detector/tree/master/output_images">output_images</a></i> folders. There are decent
  examples of input and output images.<br><br>
  If anyone is interested in UML Class Diagrams so here it is. See the comments in code for more detailed descriptions of the funtions.
</p>

![diagram2](https://user-images.githubusercontent.com/91671608/162276354-2c15ee20-a7e1-4dd5-8de3-0302751531c1.png)

