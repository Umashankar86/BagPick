# Hand Object Detector 
This is a pure pytorch and numpy implementaion of the code for *Understanding Human Hands in Contact at Internet Scale* (CVPR 2020, **Oral**).

Dandan Shan, Jiaqi Geng*, Michelle Shu*, David F. Fouhey

![method](assets/method.png)



## Introduction

This repo is the pytorch implementation of a Hand Object Detector based on Faster-RCNN. The code is based on a fork of https://github.com/ddshan/hand_object_detector but all explicit C dependencies are removed and a simple interface is added to make their awsome work easier to deploy. Pleas check out the original repo for more information.

More information can be found at our:

* [Project and dataset webpage](http://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/)


## Prerequisites

The code should with most versions of pytorch,torchvision and numpy. You will also need matplotlib, and open-cv for visualizations.


## Environment & Compilation

For convinience you can install the package using pip install -e .

## Demo

### Image Demo

**Download models** by using this link: https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE.


**Save models** in the handobdet/models/** folder:
```
mkdir models

models
└── res101_handobj_100K
    └── pascal_voc
        └── faster_rcnn_1_8_132028.pth
```



**Simple testing**: 



**Params to save detected results** in demo.py you may need for your task:
* hand_dets: detected results for hands, [boxes(4), score(1), state(1), offset_vector(3), left/right(1)]
* obj_dets: detected results for object, [boxes(4), score(1), <em>state(1), offset_vector(3), left/right(1)</em>]

We did **not** train the contact_state, offset_vector and hand_side part for objects. We keep them just to make the data format consistent. So, only use the bbox and confidence score infomation for objects.  




### One Image Demo Output:

Check out demo.py for a simple demo.

Color definitions:
* yellow: object bbox
* red: right hand bbox
* blue: left hand bbox

Label definitions:
* L: left hand
* R: right hand
* N: no contact
* S: self contact
* O: other person contact
* P: portable object contact
* F: stationary object contact (e.g.furniture)


![demo_sample](assets/boardgame_848_sU8S98MT1Mo_00013957.png)


### Limitations
- Occasional false positives with no people.
- Issues with left/right in egocentric data (Please check egocentric models that work far better).
- Difficulty parsing the full state with lots of people.

<!-- ## Acknowledgment

xxx -->

## Citation

If this work is helpful in your research, please cite:
```
@INPROCEEDINGS{Shan20, 
    author = {Shan, Dandan and Geng, Jiaqi and Shu, Michelle  and Fouhey, David},
    title = {Understanding Human Hands in Contact at Internet Scale},
    booktitle = CVPR, 
    year = {2020} 
}
```
When you use the model trained on our ego data, make sure to also cite the original datasets ([Epic-Kitchens](https://epic-kitchens.github.io/2018), [EGTEA](http://cbs.ic.gatech.edu/fpv/) and [CharadesEgo](https://prior.allenai.org/projects/charades-ego)) that we collect from and agree to the original conditions for using that data.
