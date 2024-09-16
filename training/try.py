import gradio as gr  # 导入gradio库，用于创建交云式Web应用界面
 
 
# 定义一个函数，接受一个名字作为输入，返回一个问候语
def greet(name):
    return "Hello " + name + "!"  # 拼接问候语
 
 
# 使用‘with’语法和Blocks API构造Gradio界面
with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")  # 创建一个文本框，用于输入名字
    output = gr.Textbox(label="Output Box")  # 创建一个文本框，用于显示问候语输出
    greet_btn = gr.Button("Greet")  # 创建一个按钮，点击后将触发问候语函数
    # 为按钮设置点击事件，绑定greet函数，指定输入为name文本框，输出到output文本框
    greet_btn.click(fn=greet, inputs=[name], outputs=[output], api_name="greet")
 
 
demo.launch(share=True, 
                    inbrowser=True, 
                    server_name='0.0.0.0', 
                     
                    debug=True)
