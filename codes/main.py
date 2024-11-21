import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from Unsupervised_building_extraction import generate_train_obj_file_ch as GTF
from CLIP import object_filter_tk as OF
from Unsupervised_building_extraction import unsupervised_building_extraction_ch as UBE
from Unsupervised_building_extraction import train_ch as TRAIN
from Unsupervised_building_extraction import test_v2_ch as TEST
import os
from utils.metrics import Evaluator
from PIL import Image
import numpy as np


def step1(img_path,building_ins_path, none_building_ins_path, cross_building_ins_path, cross_none_building_ins_path, infer_path):
    UBE.prepare_path(building_ins_path)
    UBE.prepare_path(none_building_ins_path)
    UBE.prepare_path(cross_building_ins_path)
    UBE.prepare_path(cross_none_building_ins_path)
    UBE.prepare_path(infer_path)
    b_mask = UBE.B_Masks(img_path, building_ins_path, none_building_ins_path, cross_building_ins_path,cross_none_building_ins_path,
                     infer_path)
    b_mask.go()

def step2(root):
    name_List = ['building_ins', 'cross_building_ins', 'none_building_ins', 'cross_none_building_ins']
    for name in name_List:
        print(' processing ', name, '...')
        source_path = root + '\\' + name
        new_building_path = root + '\\' + name + '_b'
        new_none_building_path = root + '\\' + name + '_nb'
        UBE.prepare_path(new_building_path)
        UBE.prepare_path(new_none_building_path)
        OF.find_none_building(source_path, new_building_path, new_none_building_path)

def step3(tgt_path, generate_train_file_path):
    root = tgt_path + '\\'
    if not os.path.exists(generate_train_file_path):
        os.mkdir(generate_train_file_path)
    GTF.generate_file(root, generate_train_file_path)

def step4(check_path, train_b_path, train_nb_path):
    if not os.path.exists(check_path):
        os.mkdir(check_path)

    TRAIN.train_(check_path=check_path,
           train_b_path=train_b_path,
           train_nb_path = train_nb_path,
           batch_size=32,
           epochs=200,
           train_=True)

def step5(img_path, ckp_path, predict_path):
    root_path = img_path
    if not os.path.exists(predict_path):
        os.mkdir(predict_path)

    b_mask = TEST.B_Masks(root_path, ckp_path, predict_path)
    b_mask.go()

def evaluate_performance_of_single(infer_path, lab_path):
    test_binary_evaluator = Evaluator(2)

    pred = np.array(Image.open(infer_path))
    gt = np.array(Image.open(lab_path))
    if len(gt.shape) == 3:
        gt = gt[:,:,0]
    if len(pred.shape) == 3:
        pred = pred[:,:,0]
    print(pred.shape, gt.shape)
    pred = pred / 255.0
    gt = gt / 255.0

    pred = pred[:,:gt.shape[1]]

    # add the batch to the evaluator
    test_binary_evaluator.add_batch(gt, pred)

    precision, recall, F1_class, F1 = test_binary_evaluator.F1_score()
    F1 = F1 * 100
    F1_class = [x * 100 for x in F1_class]
    precision = [x * 100 for x in precision]
    recall = [x * 100 for x in recall]
    overall_presicion = (precision[0] + precision[1]) / 2
    overall_recall = (recall[0] + recall[1]) / 2
    print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(F1, overall_presicion, overall_recall, F1_class[1], F1_class[0], precision[1], precision[0], recall[1], recall[0]))
    print('+++++++++++++++++++++++++++++++++++++++')


if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.join(os.getcwd()))
    upper_path = os.path.abspath(os.path.join(abs_path, os.pardir))

    img_path = 'E:\code\ohsome2label\SH_down\stockholm\img_rename'
    tgt_path = 'E:\code\ohsome2label\SH_down\stockholm'
    dataset_name = 'STOCKHOLM'

    generate_train_file_path = "{}/Unsupervised_building_extraction/data/{}".format(upper_path, dataset_name)
    building_ins_path = tgt_path + '\\building_ins'
    none_building_ins_path = tgt_path + '\\none_building_ins'
    cross_building_ins_path = tgt_path + '\\cross_building_ins'
    cross_none_building_ins_path = tgt_path + '\\cross_none_building_ins'
    predict_path = tgt_path  + '\\infers'
    check_path = 'outputs\\' + dataset_name
    train_b_path = 'E:\code\segment-anything-main\\Unsupervised_building_extraction\data\\' + dataset_name + '\\building.txt'
    train_nb_path = 'E:\code\segment-anything-main\\Unsupervised_building_extraction\data\\' + dataset_name + '\\none_building.txt'

    # step1(img_path, building_ins_path, none_building_ins_path, cross_building_ins_path, cross_none_building_ins_path, predict_path)
    #
    # step2(tgt_path)
    #
    # step3(tgt_path, generate_train_file_path)
    #
    # step4(check_path, train_b_path, train_nb_path)
    #
    step5(img_path, check_path, predict_path)

    # step6 evaluate
    infer_path = predict_path + "\\predict.png"
    lab_path = tgt_path + "\\lab_stitch\\label2.png"
    evaluate_performance_of_single(infer_path, lab_path)
    pseudo_label = predict_path + "\\pseudo_label.png"
    evaluate_performance_of_single(pseudo_label, lab_path)
