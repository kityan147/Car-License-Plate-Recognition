# Car-License-Plate-Recognition
Using Faster RCNN and KNN to perform Car License Plate Recognition

This project uses the Faster RCNN to localize the car license plate. Then, the localized car license plate will be cropped and resize in to larger image. By obtaining the Hue value and convert it into a binary threshold image,noise cancelling with tophat filter can easily differentiate different character. The characters recognition uses K-Nearest Neighbour (K = 1) with a pre-trained classifiction value to determine the character.

The neural network architecture can be found in the model folder. There are 3 .prototxt files.
1. train.prototxt
    - The network architecture definition with specified Input data layer for training.
    - It has 5 convolutional layers, Region Proposal Network,  Softmax with loss layer for labels and Smooth L1 loss for bounding box     prediction.
2. text.prototxt
    - The network architecture definition with non specified Input data layer for testing.
    - It has 5 convolutional layers, Region Proposal Network,  Softmax layer for class probability, which is similar to the train.prototxt.
3. solver.prototxt
    - This prototxt defines the learning parameters of the training process.
    - The learning parameters that include:
       - Base Learning Rate
       - Learning Rate Policy, like step down/exponential
       - Gamma for learning rate
       - Step size
       - Display Prompt
       - Average loss
       - Weight Decay


The demo can be performed with the demo_plate_number.py in tools folder. The following code has to be modified before it runs.
```
 prototxt = "{Path of test.prototxt}"
 caffemodel = "{Path of the pre-trained caffe model}"
 ...
 im_names = ['{Path of the desired image}']
```


# Reference
    renNIPS15fasterrcnn
    Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
    Title = {Faster {R-CNN}: Towards Real-Time Object Detection
             with Region Proposal Networks},
    Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
    Year = {2015}
    
    https://github.com/rbgirshick/py-faster-rcnn
