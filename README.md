# Deepfake System: A System of Deepfake Edit and Deepfake Detection

Welcome to *Deepfake System*.This system consists of two parts, The part of Deepfake is based on multiple rounds of interactive dialogue to complete the editing of real images, and the part of DeepfakeDetection is based on DeepfakeBench to design a UI interface.



<font size=4><b> Table of Contents </b></font>

- [Deepfake](#-features)
  - [Architecture](#1-Architecture)
  - [Usage](#2-Usage)
  - [Trouble Shooting](#3-TroubleShooting )
  - [Tips](#4-Tips)
- [DeepfakeDetection](#-DeepfakeDetection)

---


## 📚 Deepfake
<a href="#top">[Back to top]</a>




## ⏳ Deepfake

### 1. Architecture
```
gradio_web(Image preprocessing, GUI)
      |
     api       |---api---> StableDiffusion
      |        |
middleware <===|---api---> OpenAI
               |          (GPT3.5turbo/GPT4, DALLE)
               |
               |---api---> ChatGLM2-6B
```
### 2. Usage

<a href="#top">[Back to top]</a>
Run in local
1. Install requirements(`golang`,`beego` required)
   - Install golang
   - Install beego: `go install github.com/beego/bee/v2@latest`
   - Install python requirements: `pip install -r gradio_web/requirements.txt`
2. In... run `bash runner.sh` to get the whole UI interface( or `export SEG_MODEL_ENV='local' && bash runner_local.sh` if you want to run image segment model in local)

### 3. TroubleShooting

<a href="#top">[Back to top]</a>
Can't open gradio_web:
1. Check if webui.py is running.
2. Check  `$GR_PORT` using `echo $GR_PORT`.
3. If it is still unavailable and you are using it through k8s or docker, check if you successfully set the network mentioned in [Usage](#usage).
Can't connect to middleware
1. check if middleware is running.
2. Check if the `$MIDDLEWARE_ENV` in the terminal running webui matchs how middleware is running (local, k8s, docker). 
3. Check `$BEE_PORT`
Can't connect to openai
1. If error message is `Post "https://api.openai.com/v1/chat/completions": dial tcp [2a03:2880:f130:83:face:b00c:0:25de]:443: i/o timeout`, you should check if you need to set a proxy to send requests.


### 4. Tips
1.Refer to this [link](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to deploy Stable Diffusion
2.Refer to this [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) to add ControlNet
3.Refer to this [Roop](https://github.com/s0md3v/sd-webui-roop) to add Roop

## 📚 DeepfakeDetection
<a href="#top">[Back to top]</a>
## ⏳ DeepfakeDetection































## 📝 Citation

<a href="#top">[Back to top]</a>

If you find our benchmark useful to your research, please cite it as follows:

```
@inproceedings{DeepfakeBench_YAN_NEURIPS2023,
 author = {Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {4534--4565},
 publisher = {Curran Associates, Inc.},
 title = {DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf},
 volume = {36},
 year = {2023}
}
```

If interested, you can read our recent works about deepfake detection, and more works about trustworthy AI can be found [here](https://sites.google.com/site/baoyuanwu2015/home).
```
@inproceedings{UCF_YAN_ICCV2023,
 title={Ucf: Uncovering common features for generalizable deepfake detection},
 author={Yan, Zhiyuan and Zhang, Yong and Fan, Yanbo and Wu, Baoyuan},
 booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
 pages={22412--22423},
 year={2023}
}

@inproceedings{LSDA_YAN_CVPR2024,
  title={Transcending forgery specificity with latent space augmentation for generalizable deepfake detection},
  author={Yan, Zhiyuan and Luo, Yuhao and Lyu, Siwei and Liu, Qingshan and Wu, Baoyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```


## 🛡️ License

<a href="#top">[Back to top]</a>


This repository is licensed by [The Chinese University of Hong Kong, Shenzhen](https://www.cuhk.edu.cn/en) under Creative Commons Attribution-NonCommercial 4.0 International Public License (identified as [CC BY-NC-4.0 in SPDX](https://spdx.org/licenses/)). More details about the license could be found in [LICENSE](./LICENSE).

This project is built by the Secure Computing Lab of Big Data (SCLBD) at The School of Data Science (SDS) of The Chinese University of Hong Kong, Shenzhen, directed by Professor [Baoyuan Wu](https://sites.google.com/site/baoyuanwu2015/home). SCLBD focuses on the research of trustworthy AI, including backdoor learning, adversarial examples, federated learning, fairness, etc.

If you have any suggestions, comments, or wish to contribute code or propose methods, we warmly welcome your input. Please contact us at wubaoyuan@cuhk.edu.cn or yanzhiyuan1114@gmail.com. We look forward to collaborating with you in pushing the boundaries of deepfake detection.
