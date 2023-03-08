# Abstract Visual Reasoning: An Algebraic Approach for Solving Raven’s Progressive Matrices #

This is the official PyTorch code for the following CVPR 2023 paper:

**Title**: Abstract Visual Reasoning: An Algebraic Approach for Solving Raven’s Progressive Matrices.

**Abstract**: We introduce algebraic machine reasoning, a new reasoning framework that is well-suited for abstract reasoning. Effectively, algebraic machine reasoning reduces the difficult process of novel problem-solving to routine algebraic computation. The fundamental algebraic objects of interest are the ideals of some suitably initialized polynomial ring. We shall explain how solving Raven's Progressive Matrices (RPMs) can be realized as computational problems in algebra, which combine various well-known algebraic subroutines that include: Computing the Gröbner basis of an ideal, checking for ideal containment, etc. Crucially, the additional algebraic structure satisfied by ideals allows for more operations on ideals beyond set-theoretic operations.

Our algebraic machine reasoning framework is not only able to select the correct answer from a given answer set, but also able to generate the correct answer with only the question matrix given. Experiments on the I-RAVEN dataset yield an overall 93.2% accuracy, which significantly outperforms the current state-of-the-art accuracy of 77.0% and exceeds human performance at 84.4% accuracy.


## 0. Illustration

![](flowchart3.png)

## 1. Requirements

- Python 3.8
- PyTorch=1.9.1
- tqdm
- mmcv
- mmdet


## 2. Datasets 

To demonstrate the effectiveness of our algebraic machine reasoning framework, we conduct experiments on RAVEN/IRAVEN datasets. These two datasets use the same generation process of the question matrix. While I-RAVEN provides a better way to generate the answer set, which overcomes the flaw of RAVEN that the correct answer could be directly inferred via majority voting, even without the question matrix.

<img src="iraven_example.png" width="400">


## 3. Object Detection Models

In order to represent the RPM instances algebraically, we first need to train object detection models to extract the attribute values from the raw RPM images. We use the MMDetection package for standard training of a RetinaNet with ResNet-50 for attributes "type", "color", "position" and "size". 

### 3.1 Data processing
The RAVEN/I-RAVEN dataset needs to be processed to a particular format called the middle format specified for MMDetection. For example, to process data to MMDetection middle format for attributes "type", "color", "size", and "position" using .npz files with 60-20-20 split for training, validating and testing:
```
python process-data.py --data-dir *directory containing the configuration files*  
                       --label-attrib types colors abssizes abspositions 
                       --save-dir *directory to save generated jpeg images* 
                       --train 60 --val 20 --test 20
```

### 3.2 Training

The configuration files are stored in "./perception/configs/". To train the object detection model on attribute "type":

```
python train-det.py ./configs/perception-types.py
```

For other attributes, just change to the corresponding configuration file.



## 4. Algebraic Machine Reasoning