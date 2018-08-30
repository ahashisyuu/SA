import six


class test:
    def __init__(self):
        print('test')


def get(model_name):
    assert isinstance(model_name, str)
    assert callable(model_name)


module_objects = globals()








