#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pdb
import sys
from textwrap import dedent
from six import string_types
from collections import OrderedDict
from typing import Any


class MetaBase(type):
    def doprenew(cls, *args, **kwargs):
        return cls, args, kwargs

    def donew(cls, *args, **kwargs):
        _obj = cls.__new__(cls, *args, **kwargs)
        return _obj, args, kwargs

    def dopreinit(cls, _obj, *args, **kwargs):
        return _obj, args, kwargs

    def doinit(cls, _obj, *args, **kwargs):
        _obj.__init__(*args, **kwargs)
        return _obj, args, kwargs

    def dopostinit(cls, _obj, *args, **kwargs):
        return _obj, args, kwargs

    def __call__(cls, *args, **kwargs):
        cls, args, kwargs = cls.doprenew(*args, **kwargs)
        _obj, args, kwargs = cls.donew(*args, **kwargs)
        _obj, args, kwargs = cls.dopreinit(_obj, *args, **kwargs)
        _obj, args, kwargs = cls.doinit(_obj, *args, **kwargs)
        _obj, args, kwargs = cls.dopostinit(_obj, *args, **kwargs)
        return _obj


class MetaParams(MetaBase):
    def __new__(meta, name, bases, dct):
        # Remove params from class definition to avoid inheritance
        # (and hence "repetition")
        newparams = dct.pop('params', ())

        packs = 'packages'
        newpackages = tuple(dct.pop(packs, ()))  # remove before creation

        fpacks = 'frompackages'
        fnewpackages = tuple(dct.pop(fpacks, ()))  # remove before creation

        # Create the new class - this pulls predefined "params"
        cls = super(MetaParams, meta).__new__(meta, name, bases, dct)

        # Pulls the param class out of it - default is the empty class
        params = getattr(cls, 'params', AutoInfoClass)

        # Pulls the packages class out of it - default is the empty class
        packages = tuple(getattr(cls, packs, ()))
        fpackages = tuple(getattr(cls, fpacks, ()))

        # get extra (to the right) base classes which have a param attribute
        morebasesparams = [x.params for x in bases[1:] if hasattr(x, 'params')]

        # Get extra packages, add them to the packages and put all in the class
        for y in [x.packages for x in bases[1:] if hasattr(x, packs)]:
            packages += tuple(y)

        for y in [x.frompackages for x in bases[1:] if hasattr(x, fpacks)]:
            fpackages += tuple(y)

        cls.packages = packages + newpackages
        cls.frompackages = fpackages + fnewpackages

        # Subclass and store the newly derived params class
        cls.params = params._derive(name, newparams, morebasesparams)

        clsmodule = sys.modules[cls.__module__]
        newclsname = str(cls.__name__ + '_' + name)  # str - Python 2/3 compat

        # This loop makes sure that if the name has already been defined, a new
        # unique name is found. A collision example is in the plotlines names
        # definitions of bt.indicators.MACD and bt.talib.MACD. Both end up
        # definining a MACD_pl_macd and this makes it impossible for the pickle
        # module to send results over a multiprocessing channel
        namecounter = 1
        while hasattr(cls, newclsname):
            newclsname += str(namecounter)
            namecounter += 1

        newcls = type(newclsname, (cls,), {})
        setattr(clsmodule, newclsname, newcls)
        return cls

    def donew(cls, *args, **kwargs):
        clsmod = sys.modules[cls.__module__]
        # import specified packages
        for p in cls.packages:
            if isinstance(p, (tuple, list)):
                p, palias = p
            else:
                palias = p

            pmod = __import__(p)

            plevels = p.split('.')
            if p == palias and len(plevels) > 1:  # 'os.path' not aliased
                setattr(clsmod, pmod.__name__, pmod)  # set 'os' in module

            else:  # aliased and/or dots
                for plevel in plevels[1:]:  # recurse down the mod
                    pmod = getattr(pmod, plevel)

                setattr(clsmod, palias, pmod)

        # import from specified packages - the 2nd part is a string or iterable
        for p, frompackage in cls.frompackages:
            if isinstance(frompackage, string_types):
                frompackage = (frompackage,)  # make it a tuple

            for fp in frompackage:
                if isinstance(fp, (tuple, list)):
                    fp, falias = fp
                else:
                    fp, falias = fp, fp  # assumed is string

                # complain "not string" without fp (unicode vs bytes)
                pmod = __import__(p, fromlist=[str(fp)])
                pattr = getattr(pmod, fp)
                setattr(clsmod, falias, pattr)
                for basecls in cls.__bases__:
                    setattr(sys.modules[basecls.__module__], falias, pattr)

        # Create params and set the values from the kwargs
        params = cls.params()
        for pname, pdef in cls.params._getitems():
            setattr(params, pname, kwargs.pop(pname, pdef))

        # Create the object and set the params in place
        _obj, args, kwargs = super(MetaParams, cls).donew(*args, **kwargs)
        _obj.params = params
        _obj.p = params  # shorter alias

        # Parameter values have now been set before __init__
        return _obj, args, kwargs


class _Sentinel(object):
    """
        Base class for Sentinel objects.
        Construction of sentinel objects.
        Sentinel objects are used when you only care to check for object identity.
    """
    __slots__ = ('__weakref__',)

def is_sentinel(obj):
    return isinstance(obj, _Sentinel)

def sentinel(name, doc=None):
    try:
        value = sentinel._cache[name]  # memoized
    except KeyError:
        pass
    else:
        if doc == value.__doc__:
            return value

        raise ValueError(dedent(
            """\
            New sentinel value %r conflicts with an existing sentinel of the
            same name.
            Old sentinel docstring: %r
            New sentinel docstring: %r

            The old sentinel was created at: %s

            Resolve this conflict by changing the name of one of the sentinels.
            """,
        ) % (name, value.__doc__, doc, value._created_at))

    try:
        frame = sys._getframe(1)
    except ValueError:
        frame = None

    if frame is None:
        created_at = '<unknown>'
    else:
        created_at = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)

    @object.__new__   # bind a single instance to the name 'Sentinel'
    class Sentinel(_Sentinel):
        __doc__ = doc
        __name__ = name

        # store created_at so that we can report this in case of a duplicate
        # name violation
        _created_at = created_at

        def __new__(cls):
            raise TypeError('cannot create %r instances' % name)

        def __repr__(self):
            return 'sentinel(%r)' % name

        def __reduce__(self):
            return sentinel, (name, doc)

        def __deepcopy__(self, _memo):
            return self

        def __copy__(self):
            return self

    cls = type(Sentinel)
    try:
        cls.__module__ = frame.f_globals['__name__']
    except (AttributeError, KeyError):
        # Couldn't get the name from the calling scope, just use None.
        # AttributeError is when frame is None, KeyError is when f_globals
        # doesn't hold '__name__'
        cls.__module__ = None

    sentinel._cache[name] = Sentinel  # cache result
    return Sentinel


sentinel._cache = {}


class AutoInfoClass(object):
    _getpairsbase = classmethod(lambda cls: OrderedDict())
    _getpairs = classmethod(lambda cls: OrderedDict())
    _getrecurse = classmethod(lambda cls: False)

    @classmethod
    def _derive(cls, name, info, otherbases, recurse=False):
        # collect the 3 set of infos
        # info = OrderedDict(info)
        baseinfo = cls._getpairs().copy()
        obasesinfo = OrderedDict()
        for obase in otherbases:
            if isinstance(obase, (tuple, dict)):
                obasesinfo.update(obase)
            else:
                obasesinfo.update(obase._getpairs())

        # update the info of this class (base) with that from the other bases
        baseinfo.update(obasesinfo)

        # The info of the new class is a copy of the full base info
        # plus and update from parameter
        clsinfo = baseinfo.copy()
        clsinfo.update(info)

        # The new items to update/set are those from the otherbase plus the new
        info2add = obasesinfo.copy()
        info2add.update(info)

        clsmodule = sys.modules[cls.__module__]
        newclsname = str(cls.__name__ + '_' + name)  # str - Python 2/3 compat

        # This loop makes sure that if the name has already been defined, a new
        # unique name is found. A collision example is in the plotlines names
        # definitions of bt.indicators.MACD and bt.talib.MACD. Both end up
        # definining a MACD_pl_macd and this makes it impossible for the pickle
        # module to send results over a multiprocessing channel
        namecounter = 1
        while hasattr(clsmodule, newclsname):
            newclsname += str(namecounter)
            namecounter += 1

        newcls = type(newclsname, (cls,), {})
        setattr(clsmodule, newclsname, newcls)

        setattr(newcls, '_getpairsbase',
                classmethod(lambda cls: baseinfo.copy()))
        setattr(newcls, '_getpairs', classmethod(lambda cls: clsinfo.copy()))
        setattr(newcls, '_getrecurse', classmethod(lambda cls: recurse))

        for infoname, infoval in info2add.items():
            if recurse:
                recursecls = getattr(newcls, infoname, AutoInfoClass)
                infoval = recursecls._derive(name + '_' + infoname,
                                             infoval,
                                             [])

            setattr(newcls, infoname, infoval)

        return newcls

    def isdefault(self, pname):
        return self._get(pname) == self._getkwargsdefault()[pname]

    def notdefault(self, pname):
        return self._get(pname) != self._getkwargsdefault()[pname]

    def _get(self, name, default=None):
        return getattr(self, name, default)

    @classmethod
    def _getkwargsdefault(cls):
        return cls._getpairs()

    @classmethod
    def _getkeys(cls):
        return cls._getpairs().keys()

    @classmethod
    def _getdefaults(cls):
        return list(cls._getpairs().values())

    @classmethod
    def _getitems(cls):
        return cls._getpairs().items()

    @classmethod
    def _gettuple(cls):
        return tuple(cls._getpairs().items())

    def _getkwargs(self, skip_=False):
        l = [
            (x, getattr(self, x))
            for x in self._getkeys() if not skip_ or not x.startswith('_')]
        return OrderedDict(l)

    def _getvalues(self):
        return [getattr(self, x) for x in self._getkeys()]

    def __new__(cls, *args, **kwargs):
        obj = super(AutoInfoClass, cls).__new__(cls, *args, **kwargs)

        if cls._getrecurse():
            for infoname in obj._getkeys():
                recursecls = getattr(cls, infoname)
                setattr(obj, infoname, recursecls())

        return obj


class Param(object):
    """
        类实例属性赋值调用__setattr__ --- 负责在__dict__注册; 所以重载__setattr__注意, 要不手动注册__dict__
    """
    # __slots__ = ("params")

    def __init__(self, params):
        # self.params = dict(params) if isinstance(params, tuple) else params
        params = dict(params) if isinstance(params, tuple) else params
        for k, v in params.items():
            self.__dict__[k] = v

    def __setattr__(self, __name: str, __value: Any) -> None:
        raise ValueError("params is immutable")
        
    # def __set__(self, obj, value):
    #     print("entering into")
    #     raise ValueError("params is immutable")
    
    # def __get__(self, key):
    #     return self.params[key]
        
    # def __getitem__(self, key):
    #     return self.params[key]
    
    # def __setitem__(self, key, value):
    #     print("entering into __setitem__")
    #     raise ValueError("params is immutable")


class MetaBase(type):

    def __new__(cls, name, bases, attrs):
        print("Creating MetaBase class")
        params = dict(attrs.pop("params", {}))

        # clsmodule = sys.modules[cls.__module__]
        # cls.__name__ 为 MetaBase ; name为子类
        if "alias" not in params:
            newclsname = str(cls.__name__ + '_' + name)  # str - Python 2/3 compat
        else:
            newclsname = params.pop("alias")
        
        # newcls = super().__new__(cls, newclsname, bases, attrs)
        # type 与 type.__new__ 区别， 前者为新类后者为实例
        # newcls = type(newclsname, bases, attrs)
        newcls = type.__new__(cls, newclsname, bases, attrs)
        # attrs["params"] = params
        p = Param(params=params)
        setattr(newcls, "p", p)
        return newcls
    
    def __init__(cls, name, bases, dct):
        # postprocess
        cls.doinit()
        print("entering into Metabase __init__")

    # # python 对象创建两种方式 Python/C Api ; 调用对象. 如果实例对象需要定义__call__, 如果
    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        print("entering into Metabase __call__", args, kwds)
        return super().__call__(*args, **kwds)

    def doinit(cls, *args, **kwargs):
        print("entering MetaBase doinit")
        # pass 


# This is from Armin Ronacher from Flash simplified later by six
def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            # print("------d", d)
            # print("this_bases", this_bases, type(this_bases[0]), len(this_bases))
            print("entering into with_metaclass")
            # return meta(name, bases, d)
            m = meta(name, bases, d)
            # type.__new__ includes __init__
            # super().__init__(m, name, bases, d)
            return m
    return type.__new__(metaclass, "temporary_class", (), {})


# class Test(with_metaclass(MetaBase, object)):

#     params = (("alias", "test"),
#               ("meta", "metadata"))

#     def __init__(self, a) -> None:
#         print("entering into Test __init__")
#         self.a = a

#     @classmethod
#     def doinit(cls, *args, **kwargs):
#         print("entering into doinit Test")

#     def on_handle(self):
#         print("test on_handle")
#         return 5


class NodeBase(MetaBase):

    def __init__(meta, name, bases, dct):
        super().__init__(name, bases, dct)
        print("entering into NodeBase __init__")
        if "on_handle" not in dct:
            raise TypeError("NodeBase must implement on_handle method")

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        print("entering into Nodebase __call__", args, kwds)
        return super().__call__(*args, **kwds)

    pass
        

class StubBase(MetaBase):

    def __init__(meta, name, bases, dct):

        print("entering into procBase __init__")

        if "expr" not in dct:
            raise TypeError("proc must implement eval method")


class SingletonMeta(type):


    def __init__(cls, name, bases, dct):
         print("entering into SingletonMeta __init__")
         cls._instances = {}

    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        print("entering into SingletonMeta __call__")
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Test1(with_metaclass(NodeBase, object)):

    # class property
    params = (("alias", "test1"),
              ("meta", "metadata"))

    def __init__(self, a) -> None:
        print("entering into Test1 __init__")
        self.a = a

    @classmethod
    def doinit(cls, *args, **kwargs):
        print("entering into doinit Test1")
        
    def on_handle(self):
        print("test1 on_handle")
        return 5

    # 实例的__call__ --- instance()
    # def __call__(self, *args, **kwargs):
    #     print("entering into Test1 __call__")
    #     return super().__call__(*args, **kwargs)

# class Test2(with_metaclass(StubBase, object)):

#     params = (("alias", "test2"),
#               ("meta", "metadata"))

#     def __init__(self, a) -> None:
#         print("entering into Test __init__")
#         self.a = a

#     @classmethod
#     def doinit(cls, *args, **kwargs):
#         print("entering into doinit Test2")

#     def on_handle(self):
#         print("test2 on_handle")
#         return 5


class Singleton(metaclass=SingletonMeta):

    _modules = {}

    def __init__(self, a):
        print("Singleton __init__")
        self.a = 5

    # pass  # stub to allow easy subclassing without metaclasses
    def func(self):
        print("test3")


if __name__ == "__main__":

    # super issubclass / type isinstance
    # top class is object
    # metaclass 对应的attrs, 实例化子类mappingproxy
    # bases为空， 默认就是cls -- 自身
    # type --> new --> init --> __call__ --> 子类
    a = Test1(123)
    print("-----a", Test1.__name__, a.__dict__)
    a.on_handle()
    # mappingproxy
    print(a.a)
    print(a.p)
    print(a.p.__dict__)
    print(a.p.meta)
    # a.p.meta = 3

    # s = Singleton(5)
    # s1 = Singleton(6)
    # s1._modules["a"] = 3
    # print(id(s), id(s1), s1._modules, s._modules)
