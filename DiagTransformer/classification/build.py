from .diagswin_transformer import DiagSWinTransformer as diagswin

def build_model(config):
    model_type = config.MODEL.TYPE
    print(f"Creating model: {model_type}")
    if model_type == "diagswin":
        model = eval(model_type)(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DIAGSWIN.PATCH_SIZE,
            in_chans=config.MODEL.DIAGSWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.DIAGSWIN.EMBED_DIM,
            depths=config.MODEL.DIAGSWIN.DEPTHS,
            num_heads=config.MODEL.DIAGSWIN.NUM_HEADS,
            window_size=config.MODEL.DIAGSWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.DIAGSWIN.MLP_RATIO,
            qkv_bias=config.MODEL.DIAGSWIN.QKV_BIAS,
            qk_scale=config.MODEL.DIAGSWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.DIAGSWIN.APE,
            patch_norm=config.MODEL.DIAGSWIN.PATCH_NORM,
            use_shift=config.MODEL.DIAGSWIN.USE_SHIFT,
            expand_stages=config.MODEL.DIAGSWIN.EXPAND_STAGES,
            expand_sizes=config.MODEL.DIAGSWIN.EXPAND_SIZES,
            expand_layer=config.MODEL.DIAGSWIN.EXPAND_LAYER,
            diagswin_pool=config.MODEL.DIAGSWIN.DIAGSWIN_POOL,
            diagswin_stages=config.MODEL.DIAGSWIN.DIAGSWIN_STAGES,
            diagswin_windows=config.MODEL.DIAGSWIN.DIAGSWIN_WINDOWS,
            diagswin_levels=config.MODEL.DIAGSWIN.DIAGSWIN_LEVELS,
            use_conv_embed=config.MODEL.DIAGSWIN.USE_CONV_EMBED,
            use_layerscale=config.MODEL.DIAGSWIN.USE_LAYERSCALE,
            use_pre_norm=config.MODEL.DIAGSWIN.USE_PRE_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )
    return model
