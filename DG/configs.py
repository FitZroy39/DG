
MODE = '2classes'

# TOTAL_AREA_EXTENTS, AREA_EXTENTS 都是相对于ego范围的
# IMG_SIZE 占据的栅格地图大小
# DETECTION_AREA_EXTENTS 相较于栅格地图的范围

TOTAL_AREA_EXTENTS = [[-50,100], [-45, 45]]
if MODE == '2classes':
    IN_CHANNEL = 8
    log_norm = 8
    num_slice = 5
    NUM_CLASSES = 2
    CONV_STRIDE = 1
    CATEGORY_TO_ID = {
        'background': 0,
        'obstacle': 1
    }
    CLASSES_NAMES = ['background',
                     'obstacle']
    CLASSES_WEIGHT = [0.1, 10]

    SELF_CAR_EXTENTS = [[-7.18, 0], [-1.85, 1.85]]
    voxel_size = (0.5, 0.3, 3) 
    # AREA_EXTENTS = [[-5, 90], [-30, 30],
    #                 [-0.3, 2.2]]
    AREA_EXTENTS = [[-25, 60], [-40.5, 40.5],
                    [-0.3, 2.2]]

    IMG_SIZE =  (int((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) / voxel_size[0]),
      int((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) / voxel_size[1]))
      
    DETECTION_AREA_EXTENTS = [  [int((AREA_EXTENTS[0][0] - TOTAL_AREA_EXTENTS[0][0]) / voxel_size[0]), 
                            int((AREA_EXTENTS[0][1] - TOTAL_AREA_EXTENTS[0][0]) / voxel_size[0])],
                            [int((AREA_EXTENTS[1][0] - TOTAL_AREA_EXTENTS[1][0]) / voxel_size[1]), 
                            int((AREA_EXTENTS[1][1] - TOTAL_AREA_EXTENTS[1][0]) / voxel_size[1])]
                        ]

elif MODE == '10classes':
    IN_CHANNEL = 8
    SELF_CAR_EXTENTS_tuple = ((-7.18, 0), (-1.85, 1.85))
    AREA_EXTENTS_tuple = ((-50, 100), (-45, 45),
                          (-1, 2))
    SELF_CAR_EXTENTS = [[-7.18, 0], [-1.85, 1.85]]
    AREA_EXTENTS = [[-50, 100], [-50, 50],
                    [-1, 2]]  # [[-50, 100], [-50, 50], [-1, 2]]##[[-40, 97.6], [-49.6,49.6], [-0.35, 4.2]]
    voxel_size = (0.5, 0.3)##0.1
    log_norm = 4
    num_slice = 5
    IMG_SIZE = (round((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) / voxel_size[0]),
                round((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) / voxel_size[1]))
    NUM_CLASSES = 10  # 21
    CONV_STRIDE = 1
    CATEGORY_TO_ID = {
        'background': 0,
        'ray_gt': 1,
        'truck': 2,
        'car': 3,
        'pedestrian': 4,
        'pedestri': 4,
        'block': 5,
        'truck_bg': 6,
        'car_bg': 7,
        'pedestrian_bg': 8,
        'block_bg': 9,
    }
    CLASSES_NAMES = ['background',
                     'ray_gt',
                     'truck',
                     'car',
                     'pedestrian',
                     'block',
                     'truck_bg',
                     'car_bg',
                     'pedestrian_bg',
                     'block_bg']
    CLASSES_WEIGHT = [1,1,1,1,1,1,1,1,1,1]
    #CLASSES_WEIGHT = [1, 1, 1, 2, 2, 1.5, 1, 2, 2, 1.5]
    assert len(CLASSES_NAMES) == NUM_CLASSES
    assert len(CLASSES_WEIGHT) == NUM_CLASSES