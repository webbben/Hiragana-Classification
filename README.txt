Description:
------------
Neural network that learns to classify Japanese Hiragana characters.
Uses a basic image classifier model made with Tensorflow, and uses ~24000 ETL4 and ETL7 image files
to train and test the model.

read about and download ETL data here:
http://etlcdb.db.aist.go.jp/

necessary data to train this model (already listed in nn.py):
ETL4/ETL4C
ETL7/ETL7LC_1
ETL7/ETL7SC_1

Performance
-----------
The model, using 3/4 of the data to train and 1/4 of the data to test (~17900 training images, ~6000 test images)
achieves a testing accuracy of about 84% to 85%.

How to Run:
-----------
run nn.py

the etl_reader.py file is a utility file I made for reading the ETL datasets.  it includes all the functions
you need to read the data and turn it into tensorflow-readable numpy arrays. This file needs to be in the same
directory as nn.py in order for it to work correctly.

you can also use the hiragana_classifier.h5 file to load the model in tensorflow

Indepth description of process
------------------------------
First, the data is collected from the ETL files.  This took probably 75% of my time on this project due to
the data type not being commonly used and not having lots of documentation online.  I was able to refer to their specification
page and get some code to start from, but lots of experimenting, googling errors and downloading packages.
- turning data into numpy arrays
- cleaning noise from data
- standardizing data from each different ETL source
  - needed to adjust size of ETL7 images to match ETL4
- choosing data I want for network (excluding unwanted data, such as punctuation marks or obsolete hiragana)

Then, data is prepared for entry into neural network
- scaling pixel value ranges from 0-160 to 0-1 (so tensorflow can read it appropriately)
- collecting label data (number of classes, labels for each image)
- linking individual images to their label, and shuffling data
- splitting data into test and training sets

Lastly, training the model
- originally only used ETL4 data (~5000 images) which resulted in poor accuracy, ~60%
- then added one dataset from ETL7 (+9200 images), adjusted the files to be same standard as ETL4 and retrained
  - achieved ~78% accuracy, +18% from last time
- then added another dataset from ETL7 (another 9200 images), adjusted them, and got even better results
  - achieved ~85% accuracy, +7% from last time
