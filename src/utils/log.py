# -*- coding: utf-8 -*-
'''
日志记录，可同时将标准输出与标准错误输出到文件和控制台。
自带logging模块需要更改所有的print语句，此自定义模块不需要对原始程序做任何更改。
网上看应该也可以用logging改写，待研究。
'''

import sys, os

class Logger(object):
    '''
    日志记录类，支持同时将标准输出和标准错误输出到文件和控制台。
    注意：不能使用with语句，否则错误信息无法输出到文件。

    参数：
    tofile: 布尔值，是否输出到文件。
    toconsole: 布尔值，是否输出到控制台。
    filename: 字符串，日志文件路径，若tofile为False，则不需要提供。
    mode: 字符串，文件打开模式。
    kwds: 其他参数，见open。

    属性：
    stdout: 用于记录原始sys.stdout。
    stderr: 用于记录原始sys.stderr。
    file: open对象。
    '''
    def __init__(self, tofile=False, toconsole=True, filename=None, mode="a", **kwds):
        self.tofile = tofile
        self.toconsole = toconsole
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        if self.tofile:
            f = open(filename, mode=mode, **kwds)
            self.file = f
        else:
            self.file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        if self.stdout is not None:
            self.stdout.write(message)
        if self.file is not None:
            self.file.write(message)

    def flush(self):
        if self.stdout is not None:
            self.stdout.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.stderr is not None:
            sys.stderr = self.stderr
            self.stderr = None
        if self.file is not None:
            self.file.close()
            self.file = None

def test():
    log=Logger(tofile=True,toconsole=True,filename='/Users/yuez/Desktop/mylog.log',mode='a',encoding='utf-8')
    print('this is a test of INFO!')
    print('Next is a test of ERROR:')
    print(1/0)
    log.close()