# pose-estimation
Mediapipe and open-cv were used to capture the pose of the subject and attempt was made to classify the pose into three classes; attentive, intermediate and inattentive.

![image](https://user-images.githubusercontent.com/82420256/165850914-c4d7c797-0ace-47bf-a465-8e1812f2e47c.png)
Source : Mediapipe website about the landmarks identified by mediapipe

Captured pose landmarks were modified such that, each keypoint distance from the nose or 0th keypoint is used as different features for the classification task. Processed pose landmarks were used to train an SVM classifier with a linear kernel. This classifier had an accuracy of 66% on the train dataset.

This trained model is saved into a pickle file which is used to predict the classes in real-time.
