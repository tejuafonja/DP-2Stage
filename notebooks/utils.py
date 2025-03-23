import pandas as pd
def get_bs(x):
    split=x.split("/")[-5]

    if not "bs" in split:
        split=x.split("/")[-4]
    
    if not "bs" in split:
        split=x.split("/")[-3]
    
    if not "bs" in split:
        split=x.split("/")[-2]

    if "bs" in split:
        try:
            split_bs=split.split("-")[1]
            return int(split_bs.replace("bs", ""))
        except:
            return -1
        
    else:
        return -1

def get_train_ep(x):
    split=x.split("/")[-5]

    if not "-epoch" in split:
        split=x.split("/")[-4]
    
    if not "-epoch" in split:
        split=x.split("/")[-3]
    
    if not "-epoch" in split:
        split=x.split("/")[-2]

    if "-epoch" in split:
        split_ep=split.split("-")[2]
        return int(split_ep.replace("epoch", ""))
    else:
        return -1
    
def get_folder_name(x):
    split=x.split("/")[-3]
    if "synth" in split:
        return split
    else:
        return "-1"
    
def get_exp(x):
    try:
        split=x.split("runs/")[1]
        split_first=split.split("/")[0]
        if not (split_first.startswith("01") or split_first.startswith("15")) :
            return f'{split_first}_{split.split("/")[1]}'
        else:
            return f'base_{split.split("/")[1]}'
    except:
        return 'real'
    
def get_dslf(x):
    split=x["fake_path"].split('/')
    count=len(split)
    if 'baseline'in x["fake_path"]:
        xx=x['fake_path'].split("/")
        return xx[-6], xx[-7], xx[-5], -1

    if "LR" in x["fake_path"] or "2Stage" in x["fake_path"]:
        if count == 16:
            return x['fake_path'].split("/")[-6:-2] 
        elif count == 17:
            return x['fake_path'].split("/")[-7:-3] 
        elif count == 18:
            return x['fake_path'].split("/")[-8:-4] 
        elif count ==19:
            return x['fake_path'].split("/")[-9:-5] 
        else:
                return  x['fake_path'].split("/")[-3:-2] + ["real"]*3
    else:
        if count == 15:
            return x['fake_path'].split("/")[-6:-2] 
        elif count ==16:
            return x['fake_path'].split("/")[-7:-3] 
        elif count == 17:
            return x['fake_path'].split("/")[-8:-4] 
        elif count ==18:
            return x['fake_path'].split("/")[-9:-5] 
        else:
            return  x['fake_path'].split("/")[-3:-2] + ["real"]*3

def get_epoch(x):
    try:
        split=x.split("/")[-4]
        if not (split.startswith("epoch") or split.startswith("step")):
            split=x.split("/")[-3]
        if not (split.startswith("epoch") or split.startswith("step")):
            split=x.split("/")[-2]
        if split.startswith("epoch"):
            return int(split.replace("epoch",""))
        elif split.startswith("step"):
            return split
        else:
            return -1 
    except:
       return -1
        
def get_eps(x):
    try:
        if "NonDP" not in x:
            split=x.split("/")[-5]
            if "eps" not in split:
                split=x.split("/")[-4]
            if "eps" not in split:
                split=x.split("/")[-3]
            if "eps" not in split:
                split=x.split("/")[-2]

            eps_split=split.split('-')[-2]
            if eps_split.startswith("eps"):
                return int(float(eps_split.replace("eps","")))
            else:
                return "-1"
        else:
            return "nondp"
    except:
        return -1
       

def get_clip(x):
    try:
        if "NonDP" not in x:
            split=x.split("/")[-5]
            if "clip" not in split:
                split=x.split("/")[-4]
            if "clip" not in split:
                split=x.split("/")[-3]
            if "clip" not in split:
                split=x.split("/")[-2]

            
            clip_split=split.split('-')[-1]
            
            if clip_split.startswith("clip"):
                return float(clip_split.replace("clip",""))
            else:
                return -1
        
        else:

            return "nondp"
    except:
        return -1
    

def merge_metric_name(x):
    if  x['metric'] == 'efficacy_test':
        return f"{x['metric'].split('_')[0]}_{x['model_name']}_{x['scorer']}"
    elif x['metric'] == 'histogram_intersection':
        return f"{x['metric'].split('_')[0]}_{int(x['bins'])}_{x['scorer']}"
    elif x['metric'] == 'closeness_approximation' or "exact_duplicates":
        return f"{x['metric'].split('_')[0]}_{x['scorer']}"
    else:
        return ""
    

def merge_metric_name2(x):
    if  x['metric'] == 'efficacy_test':
        return f"{x['metric'].split('_')[0]}_{x['scorer']}"
    elif x['metric'] == 'histogram_intersection':
        return f"{x['metric'].split('_')[0]}_{x['scorer']}"
    elif x['metric'] == 'closeness_approximation' or "exact_duplicates":
        return f"{x['metric'].split('_')[0]}_{x['scorer']}"
    else:
        return ""