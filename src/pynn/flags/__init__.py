# define flags here
class Flags:
    __NO_GRAD = False
    __GLOBAL_TYPE = None
    __USING_CUDA = False

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
        
    @classmethod
    def using_cuda(cls):
        return cls.__USING_CUDA
    
    @classmethod
    def set_using_cuda(cls, value: bool):
        if isinstance(value, bool):
            cls.__USING_CUDA = value
        else:
            raise TypeError("value must be a boolean")