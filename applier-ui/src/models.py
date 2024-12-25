import glob
import json
import stringcase

# KEEP IT AS IT IS
def load_models(models_path: str):
    files = glob.glob(f'{models_path}/*.json')
    
    return files


def get_model_name(model_path: str) -> str:
    model_name = model_path.split('/')[-1].replace('.json', "")
    model_name = model_name.replace('-', ' ')
    return stringcase.capitalcase(model_name)


def load_model_to_dict(model_path: str):
    with open(model_path, 'r') as file:
        return json.load(file)
