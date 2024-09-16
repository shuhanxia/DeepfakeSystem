import json
import os
from PIL import Image
from model import inference
import gradio as gr
import yaml
#from model import prepare_img
import torch
from detectors import DETECTOR

# with open("config.json", 'r') as json_file:
#     config:dict = json.load(json_file)

print('config')
with open("/home/Userlist/shuhanxia/DeepfakeBench-main/training/config.json", 'r') as json_file:
    config:dict = json.load(json_file)
print(config)

#img_path = './example/example_1.jpeg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def InferenceProcess(image):
    # # TODO: Check Args:
    if image == None:
        gr.Warning("Image: None")
        return None, ''

    # TODO: implement your preprocess code
    #
    #
    with open('/home/Userlist/shuhanxia/DeepfakeBench-main/training/config/detector/ucf.yaml', 'r') as f:
         config_detector = yaml.safe_load(f)
    #print('open success')
    #test_data_loader = prepare_img(image)
    # prepare the model (detector)
    model_class = DETECTOR[config_detector['model_name']]
    #with open('/home/Userlist/shuhanxia/DeepfakeBench-main/training/config/detector/ucf.yaml', 'r') as f:
    #    config_backbone = yaml.safe_load(f)
    model = model_class(config_detector).to(device)
    epoch = 0
    weights_path = '/home/Userlist/shuhanxia/DeepfakeBench-main/training/weights/ucf_best.pth'
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    predictions = inference(model, image)
    if predictions > 0.5:
        output_text = 'True Image'
    else:
        output_text = 'Fake Image'

    # TODO: implement your code here for expected return
    #
    #

    #gr.Info("Finish Inference Process")    
    return output_text


with gr.Blocks() as demo:
    gr.HTML(f"""<h1 align="center">{"DeepFake Detection"}</h1>""")
    
    # TODO: Set Components Layout Here
    # ------------------------------------------------------------------------------------------
    with gr.Column(): # TODO: Column Layout

        # with gr.Accordion("Settings"): # TODO: Can be fold
        #     inferenceModeDropdown = gr.Dropdown(
        #         label='ExampleDropdown', 
        #         value=config['inference']['default_inference_mode'],
        #         choices=config['inference']['inference_mode_dropdown_enum']
        #         )
            
    # 为按钮设置点击事件，绑定greet函数，指定输入为name文本框，输出到output文本框
        


        with gr.Row(): # TODO: Row Layout
            with gr.Column():
                baseIMGViewer = gr.Image(label='Upload Image', type='pil', interactive=True)
                with gr.Accordion("Examples", open=False): 
                    gr.Examples(["./example/"+filename for filename in os.listdir("./example") if filename.split('.')[-1].lower() in ["jpeg","jpg","png"]], baseIMGViewer)

        inferenceBtn = gr.Button("Check", variant="primary")
        resultTextbox = gr.Textbox(label='Result Text', interactive=False)
    # ------------------------------------------------------------------------------------------


    # TODO: Bind Event Here
    # ---------------------------------------------

    inferenceBtn.click(
            #print('click success'),
            fn=InferenceProcess, 
            inputs=baseIMGViewer, 
            outputs=resultTextbox
            )
    # ---------------------------------------------


demo.queue().launch(share=False, 
                    inbrowser=True, 
                    server_name='0.0.0.0', 
                    server_port=config.get("port",27778), 
                    debug=True)