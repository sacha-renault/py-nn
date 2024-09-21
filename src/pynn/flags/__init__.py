# define flags here
class Flags:
    __NO_GRAD = False
    __GLOBAL_TYPE = None

    @classmethod
    def no_grad(cls):
        return cls.__NO_GRAD
    
    @classmethod
    def set_no_grad(cls, value: bool):
        if isinstance(value, bool):
            cls.__NO_GRAD = value
        else:
            raise TypeError("value must be a boolean")
        
    @classmethod
    def global_type(cls):
        return cls.__GLOBAL_TYPE
    
    @classmethod
    def set_global_type(cls, value):
        if isinstance(value, type):
            cls.__GLOBAL_TYPE = value
        else:
            raise TypeError("set_global_type takes a type as argument.")