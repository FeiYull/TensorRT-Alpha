import torch, os, cv2
from utils.dist_utils import dist_print
import torch, os
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # img_transforms = transforms.Compose([
    #     transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    # if cfg.dataset == 'CULane':
    #     splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
    #     datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
    #     img_w, img_h = 1640, 590
    # elif cfg.dataset == 'Tusimple':
    #     splits = ['test.txt']
    #     datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
    #     img_w, img_h = 1280, 720
    # else:
    #     raise NotImplementedError
    # for split, dataset in zip(splits, datasets):
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     print(split[:-3]+'avi')
    #     vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
    #     for i, data in enumerate(tqdm.tqdm(loader)):
    #         imgs, names = data
    #         imgs = imgs.cuda()
    #         with torch.no_grad():
    #             pred = net(imgs)

    #         vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
    #         coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
    #         for lane in coords:
    #             for coord in lane:
    #                 cv2.circle(vis,coord,5,(0,255,0),-1)
    #         vout.write(vis)
        
    #     vout.release()


    img_transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((int(cfg.train_height/cfg.crop_ratio), cfg.train_width)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
    # change to dummpy inputs
    img_path = "benchmark_velocity_test/clips/45/imgs/040.jpg"
    img = cv2.imread(img_path)
    # assert(img.all() == None)
    img_h, img_w = img.shape[0], img.shape[1]
    im0 = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[0], img.shape[1]
    img = img_transforms(img)
    img = img[:, -cfg.train_height:, :]
    img = img.to('cuda:0')
    img = torch.unsqueeze(img, 0)

    # by tensorrt-alpha
    onnx_name = ''
    if cfg.dataset == 'CULane':
        onnx_name = 'culane_dynamic.onnx' # 320 * 1600
    elif cfg.dataset == 'Tusimple':
        onnx_name = 'tusimple_dynamic.onnx' # 320 * 800
    else:
        raise NotImplementedError

    with torch.no_grad():
        pred = net(img)

        # export by tensorrt-alpha
        input_names = ["images"]
        output_names = ["output0", "output1", "output2", "output3"]
        dynamic_axes = {"images" : {0: "batch_size"}, 
                        "output0": {0: "batch_size"}, 
                        "output1": {0: "batch_size"}, 
                        "output2": {0: "batch_size"}, 
                        "output3": {0: "batch_size"}}
        torch.onnx.export(net, img, onnx_name,
                                input_names=input_names,
                                output_names=output_names,
                                verbose=True,
                                opset_version=11,
                                dynamic_axes=dynamic_axes)

        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=img_w,
                            original_image_height=img_h)
    for lane in coords:
        for coord in lane:
            cv2.circle(im0, coord, 5, (0, 255, 0), -1)
    cv2.imshow('demo', im0)
    cv2.waitKey(0)
