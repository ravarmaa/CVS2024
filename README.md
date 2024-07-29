# CVS2024


1. Create a virtual environment

2. Set up the virtual environment (Download the requirements)

3. run test_yolo.py and see if person detection works
* Make sure the device index is correct in the config.py

4. Download a dataset for face detection https://www.kaggle.com/datasets/lylmsc/wider-face-for-yolo-training?resource=download
* May require registration/login to kaggle
* Extract the archive.zip

5. Run split.py to split the dataset into training and test sets
* Point the "FACES_ARCHIVE_DIR" in the config to the extracted archive folder

6. Train yolo to detect faces - run train.py

7. Run test_yolo.py and see if face detection works