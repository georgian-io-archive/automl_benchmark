
import threading

_registry = []

def _register_class(target):
    global _registry
    _registry += [target]

def execute_methods(method):
    threads = []
    for cls in _registry:
        t = threading.Thread(target=cls.execute, args=(method,))
        threads.append(t)
        t.start()
    print("Waiting for execution of all methods to finish...")
    for t in threads: t.join()
        #cls.execute(method)

class AutoMLMethods(object):
    def __init__(self, *args):
        self.args = args
    def __call__(self, cls):
        methods = list(self.args)
        class Wrapped(cls):
            @staticmethod
            def filter(get_tests):
                filtered_tests = [x for x in get_tests() if x[1] in methods]
                return filtered_tests
        return Wrapped
                

class Register(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta,name,bases,class_dict)
        if name == 'Wrapped' : _register_class(cls)
        return cls

class Dispatcher(object, metaclass=Register):

    @classmethod
    def process(cls,tests):
        raise NotImplementedError("Function not implemented")
        
    @staticmethod
    def filter(get_tests):
        raise NotImplementedError("Function not implemented")

    @classmethod
    def execute(cls, get_tests):
        cls.process(cls.filter(get_tests))
        
