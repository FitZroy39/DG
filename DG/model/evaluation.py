import sys 
sys.path.append('../')
import utils.utils as u

from tqdm import tqdm
from prettytable import PrettyTable
import torch
import time
import numpy as np
import os

from imageio import imsave, imread
from IPython.display import clear_output


info = [('ground',''), ('drivable','')]

grouping_info = []


from skimage.transform import resize

# import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def div(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def get_result_matrix(h_lane, classes = 6):
    
    recall = np_round(div(np.diag(h_lane), h_lane.sum(1)))
    precision = np_round(div(np.diag(h_lane), h_lane.sum(0)))

    recall_thred = 0.8
    precision_thred = 0.85

    result = (precision > precision_thred) & (recall > recall_thred)
            
    return np.array(result), None

    
def evaluate(model, data_loader_test, key='Drivable_Area'):
    if key=='Drivable_Area':
        classes = 3
    elif key=="Port_Lane":
        classes = 9

    model.eval()
    confmat = u.ConfusionMatrix(classes)
    port_accu_matrix = []
    with torch.no_grad():
        for img, target in tqdm(data_loader_test):
            target = target[key].cuda()
            img = img.cuda()
            prediction = model(img)
            sem_out = torch.argmax(prediction[key], dim=1)

            confmat.update(target.flatten(), sem_out.flatten())
            if key=="Port_Lane":
                port_accu_confmat = u.ConfusionMatrix(classes)
                port_accu_confmat.update(target.flatten(), sem_out.flatten())
                port_accu_h_lane = np.array(port_accu_confmat.mat.float().cpu())
                matrix, _ = get_result_matrix(port_accu_h_lane, classes)
                port_accu_matrix.append(matrix)
    confusion_matrix = confmat.mat.float()
    confusion_matrix = np.array(confusion_matrix.cpu())
    if key=="Port_Lane":
        return confusion_matrix, np.array(port_accu_matrix)
    else:
        return confusion_matrix

def formatting_output_port2head(h_drivable, h_lane, lane_accu_matrix):
    "given confusion matrix, compute intrested info"
    # Drivable
    recall = np_round(div(np.diag(h_drivable), h_drivable.sum(1)))
    precision = np_round(div(np.diag(h_drivable), h_drivable.sum(0)))
    f1_drivable = np_round(div(2, (div(1, precision) + div(1, recall))))
    iou = np_round(div(np.diag(h_drivable), (h_drivable.sum(1)+h_drivable.sum(0)-np.diag(h_drivable))))
    
    x = PrettyTable()
    x.field_names = ['class', 'iou', 'Recall', 'precision', 'f1_score']
    x.add_row(['Drivable', iou[1], recall[1], precision[1], f1_drivable[1]])
    x.add_row(['Sea', iou[2], recall[2], precision[2], f1_drivable[2]])
    
    recall = np_round(div(np.diag(h_lane), h_lane.sum(1)))
    precision = np_round(div(np.diag(h_lane),h_lane.sum(0)))

    accu = np_round(div(lane_accu_matrix.sum(0), lane_accu_matrix.shape[0]))
    
    f1_port = np_round(div(2, (div(1, precision) + div(1, recall))))
    iou = np_round(div(np.diag(h_lane), (h_lane.sum(1) + h_lane.sum(0) - np.diag(h_lane))))
    
    y = PrettyTable()
    lane_name = ['gound', 'L2', 'L1', 'middle', 'R1', 'R2', 'Coast', 'left_curb', 'right_curb']
    y.field_names = ['class', 'iou', 'Recall', 'precision', 'f1_score', 'accu']
    
    for i in [0,1,2,3,4,5,6,7,8]:
        y.add_row([lane_name[i], iou[i], recall[i], precision[i], f1_port[i], accu[i]])
    
    drivable_accu = f1_drivable[1]
    
    # current only compute l1&r1
    accu_final = (3*accu[2] + 3*accu[4] + 2*accu[6] + 2*drivable_accu) / 10
    return np_round(accu_final, 4), x.get_string(), y.get_string()
    
def evaluate_onnx(model_path, eval_list, key='Drivable_Area'):
    
    if key=='Drivable_Area':
        classes = 2
        label_folder_name = 'Drivable_Area'
        out_idx = 0
    elif key=="Port_Lane":
        classes = 6
        label_folder_name = 'lane_multiple_classes'
        out_idx = 1
        
    ort_session = onnxruntime.InferenceSession(model_path)
    
    confmat = u.ConfusionMatrix(classes)


    for img_path in tqdm(eval_list):
#         clear_output(True)

        label_path = img_path.replace('images', label_folder_name).replace('.jpg','.png')
        
        img = imread(img_path)
        img = torch.tensor(img).permute(2,0,1).float()
        x = img.unsqueeze(0).cpu()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)

        sem_out = ort_outs[out_idx][0]
    
        target = imread(label_path)
        target = resize(target, (200, 320),
            preserve_range=True, order=0,
            mode='reflect',anti_aliasing =False).astype('uint8')

        confmat.update(torch.tensor(target).flatten(), torch.tensor(sem_out).flatten())

    confusion_matrix = confmat.mat.float()
    confusion_matrix = np.array(confusion_matrix)
    return confusion_matrix
    
def np_round(arr, decimals=4):
    return np.around(arr, decimals=decimals)

def model_speed_test(model, img, repeat = 100):
    model.eval()
    times = []
    with torch.no_grad():
        for i in tqdm(range(repeat)):
            clear_output(wait=True)
            start = time.time()
            out = model(img)
            torch.cuda.synchronize()
            times.append((time.time()-start)*1000)
    print('Speed: {} ms.'.format(np.mean(times)))

def formatting_output(h):
    precision = np_round(np.diag(h) / h.sum(0))
    recall = np_round(np.diag(h) / h.sum(1))
    iou = np_round(np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h)))
    print("mIoU: ", np.mean(iou))
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU', 'Recall', 'precision']

    for i in range(len(info)):
        x.add_row([info[i][0], iou[i], recall[i], precision[i]])

    print(x)
    
    # drivable, vehicle
    idxs = [ele[1] for ele in grouping_info]

    h_new = grouping_confusion_matrix(h, idxs)

    precision_new = np_round(np.diag(h_new) / h_new.sum(1))
    recall_new = np_round(np.diag(h_new) / h_new.sum(0))
    iou_new = np_round(np.diag(h_new) / (h_new.sum(1) + h_new.sum(0) - np.diag(h_new)))

    x = PrettyTable()
    x.field_names = ['class', 'IoU', 'Recall', 'precision']
    
    for i in range(len(grouping_info)):
        x.add_row([grouping_info[i][0], iou_new[i], recall_new[i], precision_new[i]])

    print(x)
    return np.nanmean(iou)
    
def grouping_confusion_matrix(m, idxs):
    """
        m: original confusion matrix,
        idxs: grouping info, like [[0,3,4], [5]
        return: confusion matrix in new shape
    """
    lenth = len(idxs)
    m1 = np.zeros((lenth, m.shape[0]))
    m2 = np.zeros((lenth, lenth))
    
    for i in range(lenth):
        if len(idxs[i]) == 1:
            m1[i, :] = m[idxs[i], :]
        else:
            m1[i, :] = np.sum(m[idxs[i], :], axis=0)
    for j in range(lenth):
        if len(idxs[j]) == 1:
            m2[:, j] = m1[:, idxs[j][0]]
        else:
            m2[:, j] = np.sum(m1[:, idxs[j]], axis=1)
    return m2


# def inference_on_hard_case(model, exp_name, img_paths, override=False):

#     out_path = 'test_output/' + exp_name + '/'
    
#     if not override:
#         os.mkdir(out_path)

#     for img_path in tqdm(img_paths):
#         clear_output(wait=True)
#         try:
#             img = imread(img_path)
#         except:
#             print("Didn't find image:", img_path)
#             continue
#         img = torch.tensor(img).permute(2,0,1).float().cuda()/255

#         model.eval()
#         with torch.no_grad():
#             prediction = model(img.unsqueeze(0).cuda())

#         prediction = [{key:pred[key].cpu() for key in pred} 
#                       for pred in prediction]

#         image = img.permute(1,2,0).cpu().numpy()*255

#         sem_out = torch.argmax(prediction[0]['semantic_mask'], dim=0)
#         sem_visualize = show(image/255., sem_out.cpu().numpy())

#         imsave(out_path + img_path.split('/')[-1], sem_visualize)
    