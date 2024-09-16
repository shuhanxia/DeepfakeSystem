# from sklearn import metrics
# import numpy as np


# def parse_metric_for_print(metric_dict):
#     if metric_dict is None:
#         return "\n"
#     str = "\n"
#     str += "================================ Each dataset best metric ================================ \n"
#     for key, value in metric_dict.items():
#         if key != 'avg':
#             str= str+ f"| {key}: "
#             for k,v in value.items():
#                 str = str + f" {k}={v} "
#             str= str+ "| \n"
#         else:
#             str += "============================================================================================= \n"
#             str += "================================== Average best metric ====================================== \n"
#             avg_dict = value
#             for avg_key, avg_value in avg_dict.items():
#                 if avg_key == 'dataset_dict':
#                     for key,value in avg_value.items():
#                         str = str + f"| {key}: {value} | \n"
#                 else:
#                     str = str + f"| avg {avg_key}: {avg_value} | \n"
#     str += "============================================================================================="
#     return str


# def get_test_metrics(y_pred, y_true, img_names):
#     def get_video_metrics(image, pred, label):
#         result_dict = {}
#         new_label = []
#         new_pred = []
#         # print(image[0])
#         # print(pred.shape)
#         # print(label.shape)
#         image_arr = np.array(image[:102528])
#         pred_arr = np.array(pred)
#         label_arr = np.array(label)
#         print("Image shape:", np.array(image).shape)
#         print("Pred shape:", np.array(pred).shape)
#         print("Label shape:", np.array(label).shape)
#         stacked_array = np.stack((image_arr, pred_arr, label_arr), axis=1)




#         for item in np.transpose(stacked_array, (1, 0)):
#             # 分割字符串，获取'a'和'b'的值
#             s = item[0]
#             if '\\' in s:
#                 parts = s.split('\\')
#             else:
#                 parts = s.split('/')
#             # a = parts[-2]
#             # b = parts[-1]
#             if len(parts) >= 2:
#                 a = parts[-2]
#                 b = parts[-1]
#             else:
#             # 处理索引超出范围的情况，例如记录错误或者使用默认值
#                 a = None
#                 b = parts[-1]

#             # 如果'a'的值还没有在字典中，添加一个新的键值对
#             if a not in result_dict:
#                 result_dict[a] = []

#             # 将'b'的值添加到'a'的列表中
#             result_dict[a].append(item)
#         image_arr = list(result_dict.values())
#         print(image_arr)
#         # 将字典的值转换为一个列表，得到二维数组
#         for video in image_arr:
#             print(video)
#             pred_sum = 0
#             label_sum = 0
#             leng = 0
#             for frame in video:
#                 print(frame)
#                 pred_sum += float(frame[1])
#                 print(frame[1])
#                 label_sum += int(frame[2])
#                 leng += 1
#             new_pred.append(pred_sum / leng)
#             new_label.append(int(label_sum / leng))
#         fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
#         v_auc = metrics.auc(fpr, tpr)
#         fnr = 1 - tpr
#         v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#         return v_auc, v_eer

#         # for video in image_arr:
#         #     pred_sum = 0
#         #     label_sum = 0
#         #     leng = len(video[0])
#         #     print("Length of video[0] (image paths):", len(video[0]))
#         #     print(video)
#         #     #print("Length of video[1] (predictions):", len(video[1]))
#         #     #print("Length of video[2] (labels):", len(video[2]))
#         #     for i in range(leng):
#         #         pred_sum += float(image_arr[1][i])
#         #         label_sum += int(video[2][i])
                
#         #     new_pred.append(pred_sum / leng)
#         #     new_label.append(int(label_sum / leng))
#         # fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
#         # v_auc = metrics.auc(fpr, tpr)
#         # fnr = 1 - tpr
#         # v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#         # return v_auc, v_eer


#     y_pred = y_pred.squeeze()
#     # auc
#     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
#     auc = metrics.auc(fpr, tpr)
#     # eer
#     fnr = 1 - tpr
#     eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
#     # ap
#     ap = metrics.average_precision_score(y_true, y_pred)
#     # acc
#     prediction_class = (y_pred > 0.5).astype(int)
#     correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
#     acc = correct / len(prediction_class)
#     if type(img_names[0]) is not list:
#         # calculate video-level auc for the frame-level methods.
#         v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
#     else:
#         # video-level methods
#         v_auc=auc

#     return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
# # def get_video_metrics(image, pred, label):
# #     result_dict = {}
# #     for i in range(len(image)):
# #         img = image[i]
# #         p = pred[i]
# #         l = label[i]

# #         s = img
# #         if '\\' in s:
# #             parts = s.split('\\')
# #         else:
# #             parts = s.split('/')
# #         a = parts[-2]
# #         b = parts[-1]

# #         if a not in result_dict:
# #             result_dict[a] = []

# #         result_dict[a].append((img, p, l))

# #     video_metrics = {}
# #     for key, video in result_dict.items():
# #         new_pred = []
# #         new_label = []
# #         for frame in video:
# #             new_pred.append(frame[1])
# #             new_label.append(frame[2])

# #         fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
# #         v_auc = metrics.auc(fpr, tpr)
# #         fnr = 1 - tpr
# #         v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

# #         video_metrics[key] = {'auc': v_auc, 'eer': v_eer}

# #     return video_metrics



# # def get_test_metrics(y_pred, y_true, img_names):
# #     # Calculate frame-level metrics
# #     fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
# #     auc = metrics.auc(fpr, tpr)
# #     fnr = 1 - tpr
# #     eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
# #     ap = metrics.average_precision_score(y_true, y_pred)

# #     # Calculate accuracy
# #     prediction_class = (y_pred > 0.5).astype(int)
# #     correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
# #     acc = correct / len(prediction_class)

# #     # Calculate video-level metrics
# #     if isinstance(img_names[0], list):
# #         # Video-level methods, use frame-level AUC directly
# #         v_auc = auc
# #     else:
# #         # Frame-level methods, calculate video-level metrics
# #         video_metrics = get_video_metrics(img_names, y_pred, y_true)
# #         # Calculate average video-level AUC and EER
# #         v_auc_sum = 0
# #         v_eer_sum = 0
# #         for key, metrics_dict in video_metrics.items():
# #             v_auc_sum += metrics_dict['auc']
# #             v_eer_sum += metrics_dict['eer']
# #         v_auc = v_auc_sum / len(video_metrics)
# #         v_eer = v_eer_sum / len(video_metrics)

# #     return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'video_auc': v_auc, 'label': y_true}


from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        image = np.array(image[:102528])
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
