# Deepfake System: A System of Deepfake Edit and Deepfake Detection

Welcome to *Deepfake System*.This system consists of two parts, The part of Deepfake is based on multiple rounds of interactive dialogue to complete the editing of real images, and the part of DeepfakeDetection is based on DeepfakeBench to design a UI interface.



<font size=4><b> Table of Contents </b></font>

- [UI interface](#-UI-interface)
  - [Input image](#-Input-image)
  - [Chat Box](#-Chat-Box)
  - [Choose orders](#-Choose-Orders)
  - [Edited image](#-Edited-image)
  - [DeepfakeDetection](#-DeepfakeDetection)
- [Deepfake](#-features)
  - [Architecture](#1-Architecture)
  - [Usage](#2-Usage)
  - [Trouble Shooting](#3-TroubleShooting )
  - [Tips](#4-Tips)
- [DeepfakeDetection](#-DeepfakeDetection)

---


## 📚 UI interface
<a href="#top">[Back to top]</a>
## ⏳ UI interface
### 1. Input image
![Input image]([https://github.com/user-attachments/assets/bf0f0515-cf75-46ea-8eeb-311340524940](https://github.com/user-attachments/assets/cb48e419-3fc7-437c-ae35-236b8a2be08d))

### 2. Chat Box
![Chat Box](https://github.com/user-attachments/assets/9b16eab2-b267-499a-be3a-6f7fc3860a5a)


### 3. Choose orders
![Choose orders](https://github.com/user-attachments/assets/05747988-d4a1-48a1-8439-bc404f4e9872)

### 4. Edited image


### 5. DeepfakeDetection
![DeepfakeDetection](https://github.com/user-attachments/assets/93abd865-989a-4c2c-8957-78c4514334a5)
![DeepfakeDetection Result](https://github.com/user-attachments/assets/ba0822b0-c311-4efd-8255-d0759c7f4dd5)
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

<a href="#top">[Back to top]</a><br>
Run in local<br>
1. Install requirements(`golang`,`beego` required)
   - Install golang
   - Install beego: `go install github.com/beego/bee/v2@latest`
   - Install python requirements: `pip install -r gradio_web/requirements.txt`
2. In this [link](https://github.com/shuhanxia/DeepfakeSystem/blob/main/training/multimodal-service-main/runner.sh) run `bash runner.sh` to get the whole UI interface( or `export SEG_MODEL_ENV='local' && bash runner_local.sh` if you want to run image segment model in local)

### 3. TroubleShooting

<a href="#top">[Back to top]</a><br>
Can't open gradio_web:
1. Check if webui.py is running.
2. Check  `$GR_PORT` using `echo $GR_PORT`.
3. If it is still unavailable and you are using it through k8s or docker, check if you successfully set the network mentioned in [Usage](#usage).<br>
Can't connect to middleware
1. check if middleware is running.
2. Check if the `$MIDDLEWARE_ENV` in the terminal running webui matchs how middleware is running (local, k8s, docker). 
3. Check `$BEE_PORT`<br>
Can't connect to openai
1. If error message is `Post "https://api.openai.com/v1/chat/completions": dial tcp [2a03:2880:f130:83:face:b00c:0:25de]:443: i/o timeout`, you should check if you need to set a proxy to send requests.


### 4. Tips
1.Refer to this [link](https://github.com/AUTOMATIC1111/stable-diffusion-webui) to deploy Stable Diffusion<br>
2.Refer to this [ControlNet](https://github.com/Mikubill/sd-webui-controlnet) to add ControlNet<br>
3.Refer to this [Roop](https://github.com/s0md3v/sd-webui-roop) to add Roop

## 📚 DeepfakeDetection
<a href="#top">[Back to top]</a>
## ⏳ DeepfakeDetection
1.Refer to this [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) to finish DeepfakeDetection<br>
2.In this [link](https://github.com/shuhanxia/DeepfakeSystem/blob/main/demo.py),run demo.py to get the webui of DeepfakeDetection































