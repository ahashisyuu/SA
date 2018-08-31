from models import *
from main_model.MainModel import MainModel

module_objects = globals()
module_objects.pop('MainModel')


def get(model_name):
    assert isinstance(model_name, str)
    return module_objects.get(model_name)











