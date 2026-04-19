# Cog3DMap: Multi-View Vision-Language Reasoning with 3D Cognitive Maps

<div align="center" margin-bottom="3em">

[![arXiv](https://img.shields.io/badge/arXiv-2603.23023-green)](https://arxiv.org/abs/2603.23023)
[![Website](https://img.shields.io/badge/Website-Cog3DMap-blue)](https://cog3dmap.github.io)
[![Code](https://img.shields.io/badge/Code-TBD-lightgrey)](#)

</div>

<div align="center" margin-bottom="3em">

<a target="_blank" href="https://gwakcy0.github.io/">Chanyoung Gwak</a>*,
<a target="_blank" href="https://jeongyw12382.github.io/">Yoonwoo Jeong</a>*,
<a target="_blank" href="https://rootyjeon.github.io/">Byungwoo Jeon</a>,
<a target="_blank" href="https://hyunseoklee-ai.github.io/">Hyunseok Lee</a>,
<a target="_blank" href="https://alinlab.kaist.ac.kr/shin.html">Jinwoo Shin</a>, and
<a target="_blank" href="https://cog3dmap.github.io/">Minsu Cho</a>

**POSTECH · KAIST · RLWRLD**

</div>

&nbsp;

Cog3DMap constructs an explicit 3D cognitive map from multi-view images, enabling direct spatial reasoning without relying on raw video frames. Each spatial coordinate in the map carries both semantic and geometric information, which is then fed into a Multimodal Large Language Model (MLLM) for spatial question answering and grounding.

## 📢 News

* [26.06.17] Code release (tentative)

## ✨ Architecture Overview

**(a)** Given a sequence of multi-view images, our recurrent framework progressively integrates visual observations into a unified 3D memory map. Each spatial coordinate in the map is associated with a token carrying both semantic and geometric information. 

**(b)** Then, the resulting compact and explicit 3D map is fed into the MLLM decoder for spatial reasoning.

<p align="center">
    <img src="assets/main_figure.png" width="80%"><br>
    <figcaption align="center">Overall pipeline of Cog3DMap.</figcaption>
</p>

## 🚀 Main Results Highlights

* **VSTI-Bench:** Cog3DMap-8B achieves **67.5%** average score, surpassing the previous best method by **+8.7 percentage points**.

* **VSI-Bench:** Cog3DMap-8B achieves **65.1%** average score, improving over prior state-of-the-art by **+3.9 percentage points**.

* **RoboFAC:** Competitive performance across task horizons with visual token reduction up to **90.2%**.

## ⚙️ Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Gwakcy0/Cog3DMap
    cd Cog3DMap
    ```

2. **Create a Conda environment and install dependencies:**
    We recommend using Python 3.10.
    ```bash
    conda create -n cog3dmap python=3.10
    conda activate cog3dmap
    pip install -e .
    ```

## 📊 Datasets

Cog3DMap is trained and evaluated on a variety of datasets:
* **Spatial Reasoning Instruction Tuning:**
    * [VLM-3R](https://github.com/VITA-Group/VLM-3R): 
    * [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M): Only the appearance order subset is used for training.


* **Evaluation Benchmarks:** 
    * [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench), VSTI-Bench, [RoboFAC](https://github.com/MINT-SJTU/RoboFAC/tree/main).


* **3D Scene Understanding:**
    * **3D Dense Captioning:** [Scan2Cap](https://github.com/daveredrum/Scan2Cap), using Mask3D-detected object proposals extracted from [LEO](https://github.com/embodied-generalist/embodied-generalist). 
    * **3D VQA Answering:** [ScanQA](https://github.com/ATR-DBI/ScanQA) and [Scan2cap](https://github.com/daveredrum/Scan2Cap).

## Demo
For demo results, run the following:
```Bash
python scripts/demo/demo_point3r.py
```


## 📋 Todo List

- [ ] Release model weights
- [ ] Release inference demo
- [ ] Release evaluation code and preprocessing scripts
- [ ] Release training scripts

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{gwak2025cog3dmap,
  title   = {Cog3DMap: Multi-View Vision-Language Reasoning with 3D Cognitive Maps},
  author  = {Chanyoung Gwak and Yoonwoo Jeong and Byungwoo Jeon and Hyunseok Lee and Jinwoo Shin and Minsu Cho},
  journal = {arXiv preprint arXiv:2603.23023},
  year    = {2025}
}
```

## Acknowledgements

* This work is based on [VG-LLM](https://github.com/lavi-lab/VG-LLM) by Zheng et al. and [Point3R](https://github.com/YkiWu/Point3R) by Wu et al. We thank the authors for their open-source contribution.
* This work is also built upon [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Point3R](https://github.com/), and various 3D datasets including [ScanNet](https://github.com/ScanNet/ScanNet), [ScanRefer](https://github.com/daveredrum/ScanRefer), [Scan2Cap](https://github.com/daveredrum/Scan2Cap), and [EmbodiedScan](https://github.com/OpenRobotLab/EmbodiedScan).
* We thank the developers of [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for their evaluation framework.
