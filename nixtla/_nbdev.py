# AUTOGENERATED BY NBDEV! DO NOT EDIT!

__all__ = ["index", "modules", "custom_doc_links", "git_url"]

index = {"Scaler": "data__scalers.ipynb",
         "norm_scaler": "data__scalers.ipynb",
         "inv_norm_scaler": "data__scalers.ipynb",
         "norm1_scaler": "data__scalers.ipynb",
         "inv_norm1_scaler": "data__scalers.ipynb",
         "std_scaler": "data__scalers.ipynb",
         "inv_std_scaler": "data__scalers.ipynb",
         "median_scaler": "data__scalers.ipynb",
         "inv_median_scaler": "data__scalers.ipynb",
         "invariant_scaler": "data__scalers.ipynb",
         "inv_invariant_scaler": "data__scalers.ipynb",
         "TimeSeriesDataset": "data__tsdataset.ipynb",
         "get_mask_df": "experiments_utils.ipynb",
         "TimeSeriesLoader": "data__tsloader_general.ipynb",
         "BRC": "data_datasets__business.ipynb",
         "GLB": "data_datasets__business.ipynb",
         "BusinessInfo": "data_datasets__business.ipynb",
         "Business": "data_datasets__business.ipynb",
         "SOURCE_URL": "data_datasets__tourism.ipynb",
         "NP": "data_datasets__epf.ipynb",
         "PJM": "data_datasets__epf.ipynb",
         "BE": "data_datasets__epf.ipynb",
         "FR": "data_datasets__epf.ipynb",
         "DE": "data_datasets__epf.ipynb",
         "EPFInfo": "data_datasets__epf.ipynb",
         "EPF": "data_datasets__epf.ipynb",
         "Yearly": "experiments_nbeats_sota_params__tourism.ipynb",
         "Quarterly": "experiments_nbeats_sota_params__tourism.ipynb",
         "Monthly": "experiments_nbeats_sota_params__tourism.ipynb",
         "Other": "experiments_nbeats_sota_params__m3.ipynb",
         "M3Info": "data_datasets__m3.ipynb",
         "M3": "data_datasets__m3.ipynb",
         "Weekly": "experiments_nbeats_sota_params__m4.ipynb",
         "Daily": "experiments_nbeats_sota_params__m4.ipynb",
         "Hourly": "experiments_nbeats_sota_params__m4.ipynb",
         "M4Info": "data_datasets__m4.ipynb",
         "M4": "data_datasets__m4.ipynb",
         "TourismInfo": "data_datasets__tourism.ipynb",
         "Tourism": "data_datasets__tourism.ipynb",
         "logger": "data_datasets__utils.ipynb",
         "download_file": "data_datasets__utils.ipynb",
         "Info": "data_datasets__utils.ipynb",
         "TimeSeriesDataclass": "data_datasets__utils.ipynb",
         "CrossValidationNbeats": "experiments_nbeats__cv.ipynb",
         "common_grid": "experiments_nbeats_sota_params__tourism.ipynb",
         "common_ensemble_grid": "experiments_nbeats_sota_params__tourism.ipynb",
         "M3Params": "experiments_nbeats_sota_params__m3.ipynb",
         "M4Params": "experiments_nbeats_sota_params__m4.ipynb",
         "TourismParams": "experiments_nbeats_sota_params__tourism.ipynb",
         "scale_data": "experiments_utils.ipynb",
         "train_val_split": "experiments_utils.ipynb",
         "prepare_dataset": "experiments_utils.ipynb",
         "instantiate_nbeats": "experiments_utils.ipynb",
         "model_fit_predict": "experiments_utils.ipynb",
         "evaluate_model": "experiments_utils.ipynb",
         "hyperopt_tunning": "experiments_utils.ipynb",
         "metric_protections": "losses__numpy.ipynb",
         "mae": "losses__numpy.ipynb",
         "mse": "models_esrnn_utils__evaluation.ipynb",
         "rmse": "losses__numpy.ipynb",
         "mape": "models_esrnn_utils__evaluation.ipynb",
         "smape": "models_esrnn_utils__evaluation.ipynb",
         "pinball_loss": "losses__numpy.ipynb",
         "accuracy_logits": "losses__numpy.ipynb",
         "divide_no_nan": "losses__pytorch.ipynb",
         "MAPELoss": "losses__pytorch.ipynb",
         "MSELoss": "losses__pytorch.ipynb",
         "RMSELoss": "losses__pytorch.ipynb",
         "SMAPELoss": "losses__pytorch.ipynb",
         "MASELoss": "losses__pytorch.ipynb",
         "MAELoss": "losses__pytorch.ipynb",
         "PinballLoss": "models_esrnn_utils__losses.ipynb",
         "QuadraticBarrierLoss": "losses__pytorch.ipynb",
         "MQLoss": "losses__pytorch.ipynb",
         "Chomp1d": "models_nbeats__nbeats_model.ipynb",
         "TemporalBlock": "models__component.ipynb",
         "TemporalConvNet": "models__component.ipynb",
         "ESRNN": "models_esrnn__esrnn.ipynb",
         "Batch": "models_esrnn_utils__data.ipynb",
         "Iterator": "models_esrnn_utils__data.ipynb",
         "use_cuda": "models_esrnn_utils__drnn.ipynb",
         "LSTMCell": "models_esrnn_utils__drnn.ipynb",
         "ResLSTMCell": "models_esrnn_utils__drnn.ipynb",
         "ResLSTMLayer": "models_esrnn_utils__drnn.ipynb",
         "AttentiveLSTMLayer": "models_esrnn_utils__drnn.ipynb",
         "DRNN": "models_esrnn_utils__drnn.ipynb",
         "filter_input_vars": "models_nbeats__nbeats_model.ipynb",
         "detrend": "models_esrnn_utils__evaluation.ipynb",
         "deseasonalize": "models_esrnn_utils__evaluation.ipynb",
         "moving_averages": "models_esrnn_utils__evaluation.ipynb",
         "seasonality_test": "models_esrnn_utils__evaluation.ipynb",
         "acf": "models_esrnn_utils__evaluation.ipynb",
         "Naive": "models_esrnn_utils__evaluation.ipynb",
         "SeasonalNaive": "models_esrnn_utils__evaluation.ipynb",
         "Naive2": "models_esrnn_utils__evaluation.ipynb",
         "mase": "models_esrnn_utils__evaluation.ipynb",
         "evaluate_panel": "models_esrnn_utils__evaluation.ipynb",
         "owa": "models_esrnn_utils__evaluation.ipynb",
         "evaluate_prediction_owa": "models_esrnn_utils__evaluation.ipynb",
         "LevelVariabilityLoss": "models_esrnn_utils__losses.ipynb",
         "StateLoss": "models_esrnn_utils__losses.ipynb",
         "SmylLoss": "models_esrnn_utils__losses.ipynb",
         "DisaggregatedPinballLoss": "models_esrnn_utils__losses.ipynb",
         "init_weights": "models_wnbeats__nbeats.ipynb",
         "Nbeats": "models_wnbeats__nbeats.ipynb",
         "NBeatsBlock": "models_wnbeats__nbeats_model.ipynb",
         "NBeats": "models_wnbeats__nbeats_model.ipynb",
         "IdentityBasis": "models_wnbeats__nbeats_model.ipynb",
         "TrendBasis": "models_wnbeats__nbeats_model.ipynb",
         "SeasonalityBasis": "models_wnbeats__nbeats_model.ipynb",
         "ExogenousBasisInterpretable": "models_wnbeats__nbeats_model.ipynb",
         "ExogenousBasisWavenet": "models_wnbeats__nbeats_model.ipynb",
         "ExogenousBasisTCN": "models_nbeats__nbeats_model.ipynb",
         "TCN": "models_tcn__tcn.ipynb",
         "TCNModule": "models_tcn__tcn_model.ipynb",
         "create_context": "models_wnbeats__nbeats_model.ipynb",
         "XBasisTCN": "models_wnbeats__nbeats_model.ipynb",
         "XBasisWavenet": "models_wnbeats__nbeats_model.ipynb"}

modules = ["data/scalers.py",
           "data/tsdataset.py",
           "data/tsloader_fast.py",
           "data/tsloader_general.py",
           "data/datasets/business.py",
           "data/datasets/epf.py",
           "data/datasets/m3.py",
           "data/datasets/m4.py",
           "data/datasets/tourism.py",
           "data/datasets/utils.py",
           "experiments/nbeats/cv.py",
           "experiments/nbeats/sota_params/m3.py",
           "experiments/nbeats/sota_params/m4.py",
           "experiments/nbeats/sota_params/tourism.py",
           "experiments/utils.py",
           "losses/numpy.py",
           "losses/pytorch.py",
           "models/component.py",
           "models/esrnn/esrnn.py",
           "models/esrnn/utils/data.py",
           "models/esrnn/utils/drnn.py",
           "models/esrnn/utils/esrnn.py",
           "models/esrnn/utils/evaluation.py",
           "models/esrnn/utils/losses.py",
           "models/nbeats/nbeats.py",
           "models/nbeats/nbeats_model.py",
           "models/tcn/tcn.py",
           "models/tcn/tcn_model.py",
           "models/wnbeats/nbeats.py",
           "models/wnbeats/nbeats_model.py"]

doc_url = "https://Grupo-Abraxas.github.io/nixtla/"

git_url = "https://github.com/Grupo-Abraxas/nixtla/tree/master/"

def custom_doc_links(name): return None
