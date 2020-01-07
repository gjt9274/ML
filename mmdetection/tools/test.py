import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


import random
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import matplotlib.pyplot as plt
from PIL import Image

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')

    parser.add_argument('--anno_img',default="../Image_test",help='test img folder path')
    parser.add_argument('--anno_test',default="../Anno_test",help='test annotation folder')
    parser.add_argument('-anno_test_txt',default="../core_coreless_test.txt",help='main test txt file')

    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args







def main():
    args = parse_args()
    src_txt_dir = args.anno_test
    src_img_dir = args.anno_img
    src_test_file = args.anno_test_txt

    src_xml_dir = "./data/VOCdevkit/VOC2007/Annotations"
    des_test_file = "./data/VOCdevkit/VOC2007/ImageSets/Main/test.txt"

    des_img_dir = "./data/VOCdevkit/VOC2007/JPEGImages"


    # 转成xml
    txt_list = list(sorted(os.listdir(src_txt_dir)))
    change_to_xml(txt_list,src_txt_dir,src_img_dir,src_xml_dir)

    # 图片软链接
    os.symlink(os.path.abspath(src_img_dir),os.path.abspath(des_img_dir))

    # 复制test.txt到指定路径
    shutil.copyfile(src_test_file,des_test_file)

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)

    # 生成两个txt文件
    results = pickle.load(open('./eval/result.pkl','rb'),encoding='utf-8')
    
    # test_txt = '../core_coreless_test.txt'
    
    if not os.path.exists('../predicted_file'):
        os.makedirs('../predicted_file')
        
    core_save_txt = '../predicted_file/det_test_带电芯充电宝.txt'
    coreless_save_txt = '../predicted_file/det_test_不带电芯充电宝.txt'
    
    with open(src_test_file,'r') as f:
        names = f.readlines()
    
    
    for name,result in zip(names,results):
        for core_result in result[0]:
            with open(core_save_txt,'a+') as f:
                f.write('{} {} {} {} {} {}\n'.format(name.replace('\n',''),core_result[4],core_result[0],core_result[1],core_result[2],core_result[3]))
        for coreless_result in result[1]:
            with open(coreless_save_txt,'a+') as f:
                f.write('{} {} {} {} {} {}\n'.format(name.replace('\n',''),coreless_result[4],coreless_result[0],coreless_result[1],coreless_result[2],coreless_result[3]))



def get_details(txt_name,src_txt_dir,src_img_dir):
    # load images

    img_name = txt_name.replace('txt', 'jpg')

    txt_dir = os.path.join(src_txt_dir, txt_name)
    img_dir = os.path.join(src_img_dir, img_name)

    img = Image.open(img_dir)
    img_bands = img.getbands()
    img_size = img.size
    # é€šé“
    depth = len(img_bands)
    # å®½åº¦
    width = img_size[0]
    # é«˜åº¦
    height = img_size[1]

    # è¾¹æ¡†
    boxes = []
    # ç±»åˆ«
    objects = []

    with open(txt_dir, "rb") as f:
        for line in f.readlines():
            annotation = line.split()
            boxes.append([int(x) for x in annotation[2:]])
            if annotation[1] == bytes("带电芯充电宝", encoding='utf8'):
                objects.append('core')
            else:
                objects.append('coreless')


    target = {}
    target["filename"] = img_name
    target["width"] = width
    target["height"] = height
    target["depth"] = depth
    target["boxes"] = boxes
    target["object"] = objects

    return target


def save_xml(target, target_dir):

    filename = target["filename"]
    width = target["width"]
    height = target["height"]
    depth = target["depth"]
    boxes = target["boxes"]
    objects = target["object"]

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'Image'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width

    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % depth

    for box, object in zip(boxes, objects):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = object
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % box[0]
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % box[1]
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % box[2]
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % box[3]

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    save_xml = os.path.join(target_dir, filename.replace('jpg', 'xml'))

    with open(save_xml, 'wb') as f:
        f.write(xml)




def change_to_xml(txt_list,src_txt_dir,src_img_dir,src_xml_dir):
    for item in txt_list:
        target = get_details(item,src_txt_dir,src_img_dir)
        save_xml(target,src_xml_dir)


if __name__ == "__main__":
    main()
