import json
import os
from PIL import Image
from model import inference
import gradio as gr


# with open("config.json", 'r') as json_file:
#     config:dict = json.load(json_file)

# print(config)


def InferenceProcess(img: Image.Image):
    # TODO: Check Args:
    if img == None:
        gr.Warning("Image: None")
        return None, ''

    # TODO: implement your preprocess code
    #
    #

    output_text = inference(img)

    # TODO: implement your code here for expected return
    #
    #

    gr.Info("Finish Inference Process")    
    return output_text


with gr.Blocks() as demo:
    gr.HTML(f"""<h1 align="center">{"title","demo"}</h1>""")
    
    # TODO: Set Components Layout Here
    # ------------------------------------------------------------------------------------------
    with gr.Column(): # TODO: Column Layout

        with gr.Accordion("Settings"): # TODO: Can be fold
            inferenceModeDropdown = gr.Dropdown(
                label='ExampleDropdown', 
                #value=config['inference']['default_inference_mode'],
                #choices=config['inference']['inference_mode_dropdown_enum']
                )
            
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
        InferenceProcess, 
        [baseIMGViewer, inferenceModeDropdown], 
        [resultTextbox]
        )
    # ---------------------------------------------


demo.queue().launch(share=False, 
                    inbrowser=True, 
                    server_name='0.0.0.0', 
                    server_port=config.get("port",27777), 
                    debug=True)