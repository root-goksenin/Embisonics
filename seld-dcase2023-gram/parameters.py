# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,
        finetune_mode = False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='/projects/0/prjs1338/seld/tau2020/2_1_dev_split0_accdoa_foa_model.h5',

        # INPUT PATH
        dataset_dir='/projects/0/prjs1338/seld/tau2020',

        # OUTPUT PATHS
        feat_label_dir='/projects/0/prjs1338/seld/tau2020_labels_gram',
 
        model_dir='/projects/0/prjs1338/seld/tau2020_saved_models_gram',            # Dumps the trained models and training curves in this folder
        dcase_output_dir='/projects/0/prjs1338/seld/tau2020_results_gram',    # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=32000,
        hop_len_s=0.01,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=128,

        # We do not use salsalite
        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite = 50,
        fmax_doa_salsalite = 2000,
        fmax_spectra_salsalite = 9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=20,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer
        temporal_mode = "none",
        self_attn=True,
        nb_heads=8,
        nb_self_attn_layers=2,
        
        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,              # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,

        mono_encoder='spear-base',
        inject_spatial_tokens='True', 

        learnt_token_dim=384,
        learnt_n_freq=8,

        # Mono backbone ids / paths (only the selected one is loaded)
        mono_ckpts=dict(
            gram='labhamlet/gramt-mono',
            spear_base='marcoyang/spear-base-speech-audio-v2',
            spear_large='marcoyang/spear-large-speech-audio-v2',
            atst='<ATST_frame_ckpt_path>',
        ),
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    # ===== Embisonics two-stream probe: encoder × inject sweep =====
    # --- inject ablation (GRAM mono, the three baselines) ---
    elif argv == '21':                         # "full"  = OURS
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'True'
        params['condition'] = 'full'
        params['multi_accdoa'] = False

    elif argv == '23':                         # "gram_only" = baseline (ii) mono-only
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'False'
        params['condition'] = 'gram_only'
        params['multi_accdoa'] = False

    elif argv == '22':                         # "ca_only" = baseline (iii) learnt conv tokens
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'gram'
        params['inject_spatial_tokens'] = 'Learn'
        params['condition'] = 'ca_only'
        params['multi_accdoa'] = False

    elif argv == '24':                         # "bare" = baseline (i) SELDNet from scratch
        params['dataset'] = 'foa'
        params['inject_spatial_tokens'] = 'None'   # signals: build SeldModel, no encoders
        params['condition'] = 'bare'
        params['multi_accdoa'] = False

    # --- mono-encoder sweep, all with OURS (inject='True') ---
    elif argv == '31':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-base'
        params['inject_spatial_tokens'] = 'True'
        params['multi_accdoa'] = False

    elif argv == '32':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-large'
        params['inject_spatial_tokens'] = 'True'
        params['multi_accdoa'] = False

    elif argv == '33':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'atst'
        params['inject_spatial_tokens'] = 'True'
        params['multi_accdoa'] = False

    # --- same sweep, mono-only (to isolate each backbone's own ceiling) ---
    elif argv == '41':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-base'
        params['inject_spatial_tokens'] = 'False'
        params['multi_accdoa'] = False

    elif argv == '42':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'spear-large'
        params['inject_spatial_tokens'] = 'False'
        params['multi_accdoa'] = False

    elif argv == '43':
        params['dataset'] = 'foa'
        params['mono_encoder'] = 'atst'
        params['inject_spatial_tokens'] = 'False'
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
    # feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    # params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    # params['t_pool_size'] = [feature_label_resolution, 1, 1]     # CNN time pooling
    # params['patience'] = int(params['nb_epochs'])     # Stop training if patience is reached

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14 
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13


    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
