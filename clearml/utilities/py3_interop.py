""" Convenience classes supporting python3-like concepts """


try:
    from abc import abstractclassmethod

except ImportError:

    class abstractclassmethod(classmethod):
        __isabstractmethod__ = True

        def __init__(self, callable: callable) -> None:
            callable.__isabstractmethod__ = True
            super(abstractclassmethod, self).__init__(callable)
