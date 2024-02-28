# Massively Annotated DeepFake Databases

We providing 65.3 Mio demographic and non-demographic attribute annotations of 41 different attributes for five popular DeepFake
datasets, Celeb-DF, DeepFakeDetection (DFD), FaceForensics++ (FF++), DeeperForensics-1.0 (DF-1.0), and Deepfake Detection Challenge Dataset (DFDC).

* [Research Paper (ArXiv)](https://arxiv.org/abs/2208.05845)
* [Research Paper (IEEE)](https://ieeexplore.ieee.org/document/10438899)

## Table of Contents

- [Abstract](#abstract)
- [Database Properties](#database-properties)
- [Annotated Sample Images](#annotated-sample-images)
- [Download](#download)
- [Citing](#citing)
- [Acknowledgment](#acknowledgment)
- [License](#license)


## Abstract

In recent years, image and video manipulations with Deepfake have become a severe concern for security and society. Many detection models and datasets have been proposed to detect Deepfake data reliably. However, there is an increased concern that these models and training databases might be biased and, thus, cause Deepfake detectors to fail. In this work, we investigate factors causing biased detection in public Deepfake datasets by (a) creating large-scale demographic and non-demographic attribute annotations with 47 different attributes for five popular Deepfake datasets and (b) comprehensively analysing attributes resulting in AI-bias of three state-of-the-art Deepfake detection backbone models on these datasets. The analysis shows how various attributes influence a large variety of distinctive attributes (from over 65M labels) on the detection performance which includes demographic (age, gender, ethnicity) and non-demographic (hair, skin, accessories, etc.) attributes. The results examined datasets show limited diversity and, more importantly, show that the utilised Deepfake detection backbone models are strongly affected by investigated attributes making them not fair across attributes. The Deepfake detection backbone methods trained on such imbalanced/biased datasets result in incorrect detection results leading to generalisability, fairness, and security issues. Our findings and annotated datasets will guide future research to evaluate and mitigate bias in Deepfake detection techniques.


## Database Properties
We provide massive and diverse annotations for five widely-used DeepFake detection datasets, resulting the annotation datasets  **A-Celeb-DF** (9.2M labels), **A-DFD** (4.7M labels), **A-FF++** (8.5M labels), **A-DF-1.0** (38.3M labels), and **A-DFDC** (4.6M labels). 

Existing DeepFake detection datasets contain none or only sparse annotations restricted to demographic attributes. This work provides over 65.3M labels of most 41 different attributes for five popular DeepFake detection datasets ([Celeb-DF](https://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html), [DeepFakeDetection (DFD)](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html), [FaceForensics++ (FF++)](https://github.com/ondyari/FaceForensics), [DeeperForensics-1.0 (DF-1.0)](https://github.com/EndlessSora/DeeperForensics-1.0) and [Deepfake Detection Challenge Dataset (DFDC)](https://ai.facebook.com/datasets/dfdc/)).
<img src="https://github.com/pterhoer/DeepFakeAnnotations/blob/main/A-dataset.png" width="700" height="175">


## Annotated Sample Images

Below some sample images (forged) are shown including their corresponging attribute annotations. (Top-Down: A-Celeb-DF, A-DFD, A-FF++, A-DF1.0, A-DFDC)

A positive attribute label (the person has this attribute) refers to 1, a negative attribute label (a person does not have this attribute) refers to -1, and a undefined attribute annotation is marked as 0.
<!-- ![](df_samples.png) -->
<img src="https://github.com/pterhoer/DeepFakeAnnotations/blob/main/df_samples.png" width="775" height="950">

## Download

You can download the annotations for our five datasets from [google drive](https://drive.google.com/drive/folders/1eM0TH8mEjgCz7rZT7OUW6xpHYAy83p5G?usp=sharing).


## Citing


If you use this work, please cite the following papers as well as the respective databases.

```
@ARTICLE{10438899,
  author={Xu, Ying and Terh{\"{o}}rst, Philipp and Pedersen, Marius and Raja, Kiran},
  journal={IEEE Transactions on Technology and Society}, 
  title={Analyzing Fairness in Deepfake Detection With Massively Annotated Databases}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Deepfakes;Annotations;Databases;Faces;Skin;Reliability;Feature extraction;Deepfake;Deepfake detection;Databases;Bias;Fairness;Image manipulation;Video manipulation},
  doi={10.1109/TTS.2024.3365421}}
```

```
@article{DBLP:journals/tifs/TerhorstFKDKK21,
  author    = {Philipp Terh{\"{o}}rst and
               Daniel F{\"{a}}hrmann and
               Jan Niklas Kolf and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {MAAD-Face: {A} Massively Annotated Attribute Dataset for Face Images},
  journal   = {{IEEE} Trans. Inf. Forensics Secur.},
  volume    = {16},
  pages     = {3942--3957},
  year      = {2021},
  url       = {https://doi.org/10.1109/TIFS.2021.3096120},
  doi       = {10.1109/TIFS.2021.3096120},
  timestamp = {Thu, 16 Sep 2021 18:05:24 +0200},
  biburl    = {https://dblp.org/rec/journals/tifs/TerhorstFKDKK21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@inproceedings{DBLP:conf/btas/TerhorstHKZDKK19,
  author    = {Philipp Terh{\"{o}}rst and
               Marco Huber and
               Jan Niklas Kolf and
               Ines Zelch and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {Reliable Age and Gender Estimation from Face Images: Stating the Confidence
               of Model Predictions},
  booktitle = {10th {IEEE} International Conference on Biometrics Theory, Applications
               and Systems, {BTAS} 2019, Tampa, FL, USA, September 23-26, 2019},
  pages     = {1--8},
  publisher = {{IEEE}},
  year      = {2019},
  url       = {https://doi.org/10.1109/BTAS46853.2019.9185975},
  doi       = {10.1109/BTAS46853.2019.9185975},
  timestamp = {Mon, 14 Sep 2020 18:11:03 +0200},
  biburl    = {https://dblp.org/rec/conf/btas/TerhorstHKZDKK19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



## Acknowledgment
Parts of this work was carried out during the tenure of an ERCIM ’Alain Bensoussan‘ Fellowship Programme.


## License

This project is licensed under the terms of the Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) license.
The copyright of the annotations remains with the Norwegian University of Science and Technology 2022.
