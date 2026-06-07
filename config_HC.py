class DefaultConfigs(object):
    seed = 666
    # Optimizer
    weight_decay = 5e-4
    momentum = 0.9
    init_lr = 0.001  # Adam optimizer
    
    # Training
    train_epoch = 100
    test_epoch = 1  # 单次运行用于benchmark  # 单次运行用于benchmark  # 单次运行用于benchmark
    BATCH_SIZE_TRAIN = 64
    norm_flag = True
    gpus = '0'

    # Data info
    data = 'WHU-Hi-HC'
    num_classes = 16
    patch_size = 15
    pca_components = 30
    test_ratio = 0.99  # 1% used for training

    # Model config (Interval MT: ODE -> FFN -> Multi-scale Mamba)
    depth = 3
    embed_dim = 32
    d_state = 16
    ssm_ratio = 1
    pos = False
    cls = False

    # 3DConv parameters
    conv3D_channel = 32
    conv3D_kernel_1 = (5, 5, 5)
    conv3D_kernel_2 = (7, 7, 7)
    conv3D_kernel_3 = (9, 9, 9)
    dim_patch = patch_size - conv3D_kernel_1[1] + 1  # 11
    # Calculate spatial dimensions after 3D Conv
    dim_linear_1 = (patch_size - conv3D_kernel_1[1] + 1) ** 2  # 11 * 11 = 121
    dim_linear_2 = (patch_size - conv3D_kernel_2[1] + 1) ** 2  # 9 * 9 = 81
    dim_linear_3 = (patch_size - conv3D_kernel_3[1] + 1) ** 2  # 7 * 7 = 49
    
    # paths information
    checkpoint_path = ('./' + "checkpoint/" + data + '/Interval MT_TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN)\
                      + '/PatchSize' + str(patch_size) + '_TestRatio' + str(test_ratio) \
                      + '/'  + 'Depth' + str(depth) + '_embed' + str(embed_dim) + '_dstate' + str(d_state) + '_ratio' + str(ssm_ratio)
                      + '_3Dconv' + str(conv3D_channel) + '&' + str(conv3D_kernel_1) + '&' + str(conv3D_kernel_2) + '&' + str(conv3D_kernel_3) + '/')
    logs = checkpoint_path

config = DefaultConfigs()