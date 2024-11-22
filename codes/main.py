import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from Unsupervised_building_extraction import generate_train_obj_file_ch as GTF
from CLIP import object_filter_tk as OF
from Unsupervised_building_extraction import unsupervised_building_extraction_ch as UBE
from Unsupervised_building_extraction import train_ch as TRAIN
from Unsupervised_building_extraction import test_v2_ch as TEST
import os
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

if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.join(os.getcwd()))
    upper_path = os.path.abspath(os.path.join(abs_path, os.pardir))

    img_path = 'YOUR IMAGE PATH'
    tgt_path = 'YOUR TARGET PATH'
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

    step1(img_path, building_ins_path, none_building_ins_path, cross_building_ins_path, cross_none_building_ins_path, predict_path)
    #
    step2(tgt_path)
    #
    step3(tgt_path, generate_train_file_path)
    #
    step4(check_path, train_b_path, train_nb_path)
    #
    step5(img_path, check_path, predict_path)
