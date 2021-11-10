import os


def check_cfg_exist(check_files, data, model, option):
    if "data" in check_files:
        if not os.path.exists(data):
            raise FileNotFoundError("The data cfg is not exist")
    if "model" in check_files:
        if not os.path.exists(model):
            raise FileNotFoundError("The model cfg is not exist")
    if "option" in check_files:
        if not os.path.exists(option):
            raise FileNotFoundError("The option file is not exist")


def get_superior_path(path):
    return "/".join(path.replace("\\", "/").split("/")[:-1])


def get_corresponding_cfg(model_path, check_exist=[]):
    model_dir = get_superior_path(model_path)
    data_cfg_path = os.path.join(model_dir, "data_cfg.json")
    model_cfg_path = os.path.join(model_dir, "model_cfg.json")
    option_path = os.path.join(model_dir, "option.pkl")
    if len(check_exist) > 0:
        check_cfg_exist(check_exist, data_cfg_path, model_cfg_path, option_path)
    return model_cfg_path, data_cfg_path, option_path


def get_option_path(m_path):
    return os.path.join(get_superior_path(m_path), "option.pkl")
