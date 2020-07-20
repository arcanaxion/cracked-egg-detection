# Computer Vision System to Detect Cracked Eggs in Egg Tray

There are 3 test images located in the `image` folder named `test_image1`, `test_image2` and `test_image3`. 
Change the `cv2.imread()` function parameter to the image you wish to use in the system. By default, `test_image1` is read. 
Simply run the code to use the system. 

Comments marked with a `# PLOT` indicates that the code snippet can be commented out to plot a figure of some code done in that section.

To use the adaptive threshold method for abnormality detection, change the second parameter of the `abnormal` function call in Section 4 to `2` instead of the current `1`. 
Also change the tuple unpacking in Section 5 to the `adaptive_thres` variable instead from the current `canny_thres`.