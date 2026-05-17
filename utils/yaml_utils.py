import os, sys
import shutil
from addict import Dict
import yaml
from ruamel.yaml import YAML
import ruamel.yaml as ryaml
from ruamel.yaml.scalarstring import PlainScalarString   
def read_yaml(fpath=None): 
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

def update_config_from_options(config, options):
    if options is None:
        return config
    for option in options:
        if '=' not in option:
            continue
        key, value = option.split('=', 1)
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        
        target_key = keys[-1]
        if isinstance(d[target_key], dict):
            d[target_key] = value
        else:
            try:
                d[target_key] = type(d[target_key])(value)
            except:
                d[target_key] = value
    return config

def change_yaml_by_options(yaml_path, options):
    Yaml = YAML(typ='rt')
    Yaml.width = 4096                              
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = Yaml.load(file)

    if options is not None:
        for option in options:
            if '=' not in option:
                continue
            key, value = option.split('=', 1)
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]

            target_key = keys[-1]
            existing = d.get(target_key)
            if isinstance(existing, (dict, ryaml.comments.CommentedMap)):
                d[target_key] = value
            elif isinstance(existing, str):
                d[target_key] = PlainScalarString(value)
            else:
                try:
                    d[target_key] = type(existing)(value)
                except:
                    d[target_key] = PlainScalarString(value)

    with open(yaml_path, 'w', encoding='utf-8') as file:
        Yaml.dump(config, file)

def save_yaml(args, yaml_path, options):
    dst_path = os.path.join(args.Logs.now_log_dir, os.path.basename(yaml_path))
    shutil.copyfile(yaml_path, dst_path)
    if options is not None:
        change_yaml_by_options(dst_path, options)