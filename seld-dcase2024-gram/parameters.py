# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',

        dataset_dir='/projects/0/prjs1338/seld/starss2024/',


        feat_label_dir='/projects/0/prjs1338/seld/starss2024/seld_feat_label/',

        model_dir='/projects/0/prjs1338/seld/starss2024_gram_models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='/projects/0/prjs1338/seld/starss2024_gram_result',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=32000,
        hop_len_s=0.01,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=128,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual'
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=20,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,  # Train for maximum epochs
        lr=1e-3,
        inject_spatial_tokens="True",

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses frame-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached
    params['feature_sequence_length'] = 200
    
    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2024' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
