This is a Program that trains the resnet50 model to create an image detection AI from iphone photos. I am using it to create face detection for my face specifically, almost like face id; which was actually my inspiration of making this.

To replicate on your machine, you must have all the import files on your system (pytorch, timm, numpy, pandas, matplotlib, and pillow)

You then want to created a folder called "resources", inside that folder you want 3 folders named "testData" "trainData" and "validData", inside, you want a folder of images you want the model to recognize. Each folder here should be a classification of different photos, so a folder would contain pictures of person A, and another folder here would contain pictures of person B.

The structure should look like this visually

resources
    |
    |
    |-------- testData
    |             |---------- Person A
    |             |---------- Person B ....
    |
    |-------- trainData
    |             |---------- Person A
    |             |---------- Person B ....
    |
    |-------- validData
                  |---------- Person A
                  |---------- Person B ....

I would Reccomend looking over the code and making sure you change things that would be different to your current system and run.

Some things to change are num_classes in line 42 and line 82. This should be the amount of Persons you have. So if you have person A and person B, it should be 2. 

Another thing to change is the transforms.resize() on line 60 and line 152 inputted tuple to whatever size your images are or to what size you want them to be resized to. Be careful though since less or more pixels may distort the image. You may have to decide between training performance or quality

If you decide to use different folder names, be sure to change the folder paths from lines 65 to 67

also change the number of epochs you want to do during training on line 77. Currently set to 10

you may want to change the learning rate "lr" in line 86

