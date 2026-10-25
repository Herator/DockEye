# Classification

for our project to "se" or predict if there is a pearson or a boat in the frame. The group belives that it is most reasonable to learn up an model to do this instead of relaying on zones and background substractions alone. 

## Model Discussion
when going into this project, we had little to no experiance on building models. So we needed to figure out what we needed and how we wanted it to work. First we knew we needed data, so we serch for free photos that we chould use to learn this models on and consider using [boat](https://universe.roboflow.com/aqhere/boat-detect-hsxmm) and [human](https://universe.roboflow.com/lalaland/fishing-recognition-gv5jm) from universe roboflow with our own media since we belive that it might be insufficent to learn our model on our videos when we dont have so much of it. 

The next step is considering if we going to have two models where one is only learned to look for persons and one for only boats or if one to do both? 

Therefor we think we are goin to perhaps to do both to check for best results. 

Earlier in this assignment we made zones and bacground subsstraction for motion detection and i am thinking on using them again to simplify the detection of boats and humans to erase false positive if that is possible. (but as mention, no real experiance on making such models so we need to check.)


## First implementation

in the begining we startet with 60 photos of background and 60 images of boats. 

