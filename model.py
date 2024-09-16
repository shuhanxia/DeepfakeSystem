import time
import logging
from functools import wraps
from PIL import Image, ImageOps
from ui_test import load
import torch
import torch.utils.data

from training.test import ui_test_one_dataset
from training.detectors import DETECTOR
import yaml


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tempImagePath = './temp/uploadedImg.png'

def get_answer(img: Image.Image):
    
    with open('/home/Userlist/shuhanxia/DeepfakeBench-main/training/config/detector/ucf.yaml', 'r') as f:
        config1 = yaml.safe_load(f)
    test_image = load(tempImagePath,config1)
    
    with open('/home/Userlist/shuhanxia/DeepfakeBench-main/training/config/est_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_image, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                #collate_fn=test_set.collate_fn,
                drop_last=False
            )
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    predictons = ui_test_one_dataset(model, test_data_loader)
    return predictons

    #predict_answer = test_epoch(model, test_data_loaders)

@log_inference
def inference(img: Image.Image):
    result =get_answer(tempImagePath)
    return result