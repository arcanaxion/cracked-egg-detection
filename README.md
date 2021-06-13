# Computer Vision System to Detect Cracked Eggs in Egg Tray

CSC3014 Computer Vision assignment

The purpose of this script is to process a birds-eye view image of a tray of eggs. The script produces a new image with **red rectangles indicating high certainty of cracks** and **yellow rectangles indicating moderate certainty of cracks**. 

## UPDATE - **A year later (2021-06-13)**

I have added a notebook (`cracked-egg-detection.ipynb`) for some visualization.

No modifications were made to the script — it is still the same code that was submitted for my assignment. I just added this for fun.

## Testing the `main.py` script

To test the system, create a folder named `images` in the cloned repo folder. Add [these 3 images](https://lensdump.com/a/jrR80) into the `images` folder. Ensure that the files are properly named `test_image1`, `test_image2` and `test_image3`.

There are now 3 test images located in the `images` folder.

Change the `cv2.imread()` function parameter to the image you wish to use in the system. By default, `test_image1` is read.

Simply run the code to use the system.

Comments marked with a `# PLOT` indicates that the code snippet can be commented out to plot a figure of some code done in that section.

To use the adaptive threshold method for abnormality detection, change the second parameter of the `abnormal` function call in Section 4 to `2` instead of the current `1`.

Also change the tuple unpacking in Section 5 to the `adaptive_thres` variable instead from the current `canny_thres`.