
### Multi-Person Extreme Motion Prediction

Implementation for paper
Wen Guo, Xiaoyu Bie, Xavier Alameda-Pineda, Francesc Moreno-Noguer,
[Multi-Person Extreme Motion Prediction](https://arxiv.org/abs/2105.08825), CVPR2022. 

[[paper](https://arxiv.org/abs/2105.08825)] [[Project page](https://team.inria.fr/robotlearn/multi-person-extreme-motion-prediction/)]


---
### Dependencies
This repo been tested on CUDA9, Python3.6, Pytorch1.5.1.

---
### Directory

```
ROOT
|-- datasets
    |-- pi
        |-- acro1
        `-- acro2
|-- run_exps
|-- main
|-- model
|-- utils 
|-- checkpoint
    |--pretrain_ckpt
|-- tensorboard
`-- outputs
```

---
### Preparing data
Please request and download data from [ExPI](https://zenodo.org/record/5578329#.YbjaLPHMK3J) and put the data in /datasets.

*Note: If you are NOT affiliated with an institution from a country offering an adequate level of data protection 
(most countries without EU, please check [the list](https://ec.europa.eu/info/law/law-topic/data-protection/international-dimension-data-protection/adequacy-decisions_en)), you have to sign the "Standard Contractual Clauses" when applying for the data. Please follow the instructions in the downloading website.*

---
### Test on our pretrained models
* Please download pretrained models from [model](https://drive.google.com/drive/folders/1TVgaVi_SaQl9j_KoGaa3XZb5XjGC7e0Y?usp=sharing)
and put them in ./checkpoint/pretrain_ckpt/.
* Run ./run_exps/run_pro1.sh to test on Common-Action-Split. 
(To test on Single-Action-Split and Unseen-Action-Split, please run run_pro2.sh and run_pro3.sh respectively.)

---
### Training and testing
* To train/test on Common-Action-Split, please look at ./run_exps/run_pro1.sh and uncommand the corresponding lines.
* When testing, '--save_result' option could be used to save the result of different experiments in a same file ./outputs/results.json. 
Than ./outputs/write_results.py could be used to easily generate the result table as shown in our paper.
* Same for Single-Action-Split/Unseen-Action-Split.

---
### Citing
If you find our code or data helpful, please cite our work

@article{guo2021multi,
    title={Multi-Person Extreme Motion Prediction}, 
    author={Wen,Guo and Xiaoyu, Bie and Xavier, Alameda-Pineda, Francesc,Moreno-Noguer}, 
    journal={arXiv preprint arXiv:2105.08825}, 
    year={2021} }

---
### Acknowledgments
Some codes are adapted from [HisRepItself](https://github.com/wei-mao-2019/HisRepItself).

---
### Licence
GPL


