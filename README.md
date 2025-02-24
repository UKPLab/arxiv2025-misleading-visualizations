# Protecting MLLMs against misleading visualizations

[![License](https://img.shields.io/github/license/UKPLab/ukp-project-template)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.9-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)

This repository contains the implementation of the arxiv preprint: Protecting Multimodal LLMs against misleading visualizations. The code is released under an **Apache 2.0** license.

Contact person: [Jonathan Tonglet](mailto:jonathan.tonglet@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions. 


<p align="center">
  <img width="80%" src="assets/accuracy.pdf" alt="header" />
</p>

## Abstract 

We assess the vulnerability of multimodal large language models to misleading visualizations - charts that distort the underlying data  using techniques such as truncated or inverted axes, leading readers to draw inaccurate conclusions that may support misinformation or conspiracy theories. Our analysis shows that these distortions severely harm multimodal large language models, reducing their question-answering accuracy by up to 34.8 percentage points compared to non-misleading visualizations and lowering it to the level of the random baseline. To mitigate this vulnerability, we introduce six inference-time methods to improve performance of MLLMs on misleading visualizations while preserving their accuracy on non-misleading ones. The most effective approach involves (1) extracting the underlying data table and (2) using a text-only large language model to answer questions based on the table. This method improves performance on misleading visualizations by 15.4 to 19.6 percentage points.

## Environment

Follow these instructions to recreate the environment used for all our experiments.

```
$ conda create --name misviz python=3.9
$ conda activate misviz
$ pip install -r requirements.txt
```

## Datasets

<p align="center">
  <img width="70%" src="assets/real_world_examples.pdf" alt="header" />
</p>

### Prepare the datasets

- CALVI
  - dataset introduced by Get el. (2023) in "CALVI: Critical Thinking Assessment for Literacy in Visualizations".
  - Ready to use 
  - License: CC-BY 4.0

- Real-world
  - dataset introduced in this work, based on visualizations collected by Lo et al. (2022) in "Misinformed by visualization: What do we learn from misinformative visualizations?"
  - Images should be downloaded using the script below
  - License for the QA pairs: CC-BY-SA 4.0

- CHARTOM
  - dataset introduced by Bharti et al. (2024) in "CHARTOM: A Visual Theory-of-Mind Benchmark for Multimodal Large Language Models"
  - Please contact the authors to get access to the dataset. Then, run the script below to process the dataset.

- VLAT 
  - dataset introduced by Lee et al. (2017) in "VLAT: Development of a Visualization Literacy Assessment Test"


The following script will prepare the datasets, including downloading the real-world images.

```
$ python src/dataset_preparation.py
```



## Quick start

### Evaluate a multimodal LLM on one or more dataset



```
$ python src/question_answering.py --datasets calvi-chartom-real_world-vlat --model internvl2.5/8B/
```


### Generate metadata (table, axis)

```
$ python src/chart2metadata.py --datasets calvi-chartom-real_world-vlat --model internvl2.5/8B/
```


### Redraw visualization

```
$ python src/table2code.py --datasets calvi-chartom-real_world-vlat --model qwen2.5/7B/
```


### Evaluation

Finally, evaluate the accuracy of the models

```
$ python src/evaluate.py --results_folder results_qa --output_file results_qa.csv
```


## Citation

If you find this work relevant to your research or use this code in your work, please cite our paper as follows:

```bibtex 
@article{tonglet2025misleadingvisualizations,
  title={Protecting multimodal LLMs against misleading visualizations},
  author={Tonglet, Jonathan and Marie-Francine Moens and Tinne Tuytelaars and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2502.XXXX},
  year={2025}
}
```

Furthermore, if you use the CALVI and/or the real-world dataset, please cite:

```bibtex 
@inproceedings{10.1145/3544548.3581406,
author = {Ge, Lily W. and Cui, Yuan and Kay, Matthew},
title = {CALVI: Critical Thinking Assessment for Literacy in Visualizations},
year = {2023},
isbn = {9781450394215},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3544548.3581406},
doi = {10.1145/3544548.3581406},
articleno = {815},
numpages = {18},
keywords = {Information visualization, Measurement, Psychometrics, Visualization literacy, Visualization misinformation},
location = {Hamburg, Germany},
series = {CHI '23}
}
```

```bibtex 
@inproceedings{lo2022misinformed,
  title={Misinformed by visualization: What do we learn from misinformative visualizations?},
  author={Lo, Leo Yu-Ho and Gupta, Ayush and Shigyo, Kento and Wu, Aoyu and Bertini, Enrico and Qu, Huamin},
  booktitle={Computer Graphics Forum},
  volume={41},
  number={3},
  pages={515--525},
  year={2022},
  organization={Wiley Online Library},
  doi={10.1111/cgf.14559},
  url={https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14559}
}
```

If you use the CHARTOM dataset, please cite:

```bibtex 
@article{bharti2024chartom,
  title={CHARTOM: A Visual Theory-of-Mind Benchmark for Multimodal Large Language Models},
  author={Bharti, Shubham and Cheng, Shiyun and Rho, Jihyun and Rao, Martina and Zhu, Xiaojin},
  journal={arXiv preprint arXiv:2408.14419},
  year={2024},
  volume={abs/2408.14419},
  url={https://arxiv.org/abs/2408.14419}
}
```

If you use the VLAT dataset, please cite:

```bibtex 
@article{7539634,
  author={Lee, Sukwon and Kim, Sung-Hee and Kwon, Bum Chul},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={VLAT: Development of a Visualization Literacy Assessment Test}, 
  year={2017},
  volume={23},
  number={1},
  pages={551-560},
  keywords={Data visualization;Reliability;Instruments;Bars;Conferences;Market research;Psychology;Visualization Literacy;Assessment Test;Instrument;Measurement;Aptitude;Education},
  doi={10.1109/TVCG.2016.2598920},
  url={https://ieeexplore.ieee.org/abstract/document/7539634}
}
```


## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
