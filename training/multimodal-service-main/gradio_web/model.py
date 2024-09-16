import time
import logging
from functools import wraps
from PIL import Image, ImageOps
import torch.utils.data.dataloader
from ui_test import load
import torch
import torch.utils.data
import io
import sys
import os
# from test import ui_test_one_dataset
# from test import test_epoch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
#from test import inference
from detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import yaml
from flask import Flask, request, jsonify
import os
from tqdm import tqdm

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tempImagePath = './temp/uploadedImg.png'

# 新增路由用于处理文件上传
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if file:
        # 保存上传的文件到临时路径
        img = Image.open(io.BytesIO(file.read()))
        img.save(tempImagePath)
        return jsonify({'message': 'File successfully uploaded'}), 200
    else:
        return jsonify({'error': 'Upload failed'}), 500

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def log_inference(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        logging.info(f"\n\n--------------------------------------INFERENCE--------------------------------------")
        logging.info(f"Calling function '{func.__name__}' with arguments:")
        logging.info(f"  args: {args}")
        logging.info(f"  kwargs: {kwargs}")
        
        try:
            # 调用被修饰的函数
            result = func(*args, **kwargs)
            
            # 记录函数返回的信息
            logging.info(f"Function '{func.__name__}' returned: {result}")
        except Exception as e:
            # 记录异常信息
            logging.error(f"Function '{func.__name__}' raised an error: {e}")
            raise
        finally:
            # 记录函数执行的时间
            end_time = time.time()
            logging.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tempImagePath = './save_data'



# def prepare_img(image:Image.Image):
#     with open('/home/Userlist/shuhanxia/DeepfakeBench-main/training/config/detector/ucf.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     test_image = DeepfakeAbstractBaseDataset(
#         config=config,
#         mode='test', 
#     )
    
#     # with open('/home/Userlist/shuhanxia/DeepfakeBench-main/training/config/est_config.yaml', 'r') as f:
#     #     config = yaml.safe_load(f)
#     test_data_loader = \
#             torch.utils.data.DataLoader(
#                 dataset=test_image, 
#                 batch_size=1,
#                 shuffle=False, 
#                 num_workers=0,
#                 collate_fn=test_image.collate_fn,
#                 drop_last=False
#             )
#     return test_data_loader

def get_answer(model, image:Image.Image):

    # for i, data_dict in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
    #     # get data
    #     data, label, mask, landmark = \
    #     data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
    #     label = torch.where(data_dict['label'] != 0, 1, 0)
    #     # move data to GPU
    #     data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
    #     if mask is not None:
    #         data_dict['mask'] = mask.to(device)
    #     if landmark is not None:
    #         data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
    #    predictions = infer(model, data_dict)

    with open('/home/Userlist/shuhanxia/DeepfakeBench-main/training/config/detector/ucf.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    image_tensors, landmark_tensors, mask_tensors = load(image, config)
    data_dict = {
        'image': image_tensors, 
        'label': torch.tensor(0)
    }
    data = data_dict['image']
    data = data[0]
    data = data.unsqueeze(0)
    data_dict['image'] = data.to(device)

    label = data_dict['label']
    label = torch.where(data_dict['label'] != torch.tensor(0), torch.tensor(1), torch.tensor(0))
    label = label.reshape(1)
    data_dict['label'] = label.to(device)
    #print(data_dict)
    predictions = infer(model, data_dict)
    return predictions

    #predict_answer = test_epoch(model, test_data_loaders)
@torch.no_grad()
def infer(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions

@log_inference
def inference(model, image):
    # set model to eval mode
    model = model.to(device)
    model.eval()

    # testing for all test data
    # keys = test_data_loader.keys()
    # for key in keys:
    #     data_dict = test_data_loader[key].dataset.data_dict
        # compute loss for each dataset
    predictions = get_answer(model, image)
    predictions = predictions['prob']
    predictions = predictions.item()
    predictions = int(predictions)
        
        # compute metric for each dataset
        # metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
        #                                       img_names=data_dict['image'])
        # metrics_all_datasets[key] = metric_one_dataset
        
        # # info for each dataset
        # tqdm.write(f"dataset: {key}")
        # for k, v in metric_one_dataset.items():
        #     tqdm.write(f"{k}: {v}")

    return predictions