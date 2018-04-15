# 结构模式
# 1. 适配器
# 2. 修饰器
# 3. 外观
# 4. 享元
# 5. MVC
# 6. 代理

class Synthesizer:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return 'the {} synthesizer'.format(self.name)
    
    def play(self):
        return 'is playing an electronic song'

class Human:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return '{} the human'.format(self.name)
    
    def speak(self):
        return 'says hello'

from external import Synthesizer, Human
class Computer:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return 'the {} computer'.format(self.name)
    
    def execute(self):
        return 'executes a program'

class Adapter:
    def __init__(self, obj, adapted_methods):
        self.obj = obj
        self.__dict__.update(adapted_methods)

    def __str__(self):
        return str(self.obj)

def main():
    objects = [Computer('Asus')]
    synth = Synthesizer('moog')
    objects.append(Adapter(synth, dict(execute=synth.play)))
    human = Human('Bob')
    objects.append(Adapter(human, dict(execute=human.speak)))

    for i in objects:
        print('{} {}'.format(str(i), i.execute()))

if __name__ == "__main__":
    main()


# ---修饰器模型
import functools

def memoize(fn):
    known = dict()

    @functools.wraps(fn)
    def memoizer(*args):
        if args not in known:
            known[args] = fn(*args)
        return known[args]
    return memoizer

@memoize
def nsum(n):
    '''返回前n个数字的和'''
    assert(n >= 0), 'n must be >= 0'
    return 0 if n == 0 else n + nsum(n-1)

@memoize
def fibonacci(n):
    '''返回斐波那契数列的第n个数'''
    assert(n >= 0), 'n must be >= 0'
    return n if n in (0, 1) else fibonacci(n-1) + fibonacci(n-2)

if __name__ == '__main__':
    from timeit import Timer
    measure = [ {'exec':'fibonacci(100)', 'import':'fibonacci',
    'func':fibonacci},{'exec':'nsum(200)', 'import':'nsum',
    'func':nsum} ]
    for m in measure:
        t = Timer('{}'.format(m['exec']), 'from __main__ import
        {}'.format(m['import']))
        print('name: {}, doc: {}, executing: {}, time:
        {}'.format(m['func'].__name__, m['func'].__doc__,
        m['exec'], t.timeit()))


# --- 外观模式 ---
from enum import Enum
from abc import ABCMeta, abstractmethod

State = Enum('State', 'new running sleeping restart zombie')

class User:
    pass

class Process:
    pass

class File:
    pass

class Server(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    def __str__(self):
        return self.name
    
    @abstractmethod
    def boot(self):
        pass
    
    @abstractmethod
    def kill(self, restart=True):
        pass

class FileServer(Server):
    def __init__(self):
        '''初始化文件服务进程要求的操作'''
        self.name = 'FileServer'
        self.state = State.new

    def boot(self):
        print('booting the {}'.format(self))
        '''启动文件服务进程要求的操作'''
        self.state = State.running
    
    def kill(self, restart=True):
        print('Killing {}'.format(self))
        '''终止文件服务进程要求的操作'''
        self.state = State.restart if restart else State.zombie
    
    def create_file(self, user, name, permissions):
        '''检查访问权限的有效性、用户权限等'''
        print("trying to create the file '{}' for user '{}' with permissions
        {}".format(name, user, permissions))


class ProcessServer(Server):
    def __init__(self):
        '''初始化进程服务进程要求的操作'''
        self.name = 'ProcessServer'
        self.state = State.new

    def boot(self):
        print('booting the {}'.format(self))
        '''启动进程服务进程要求的操作'''
        self.state = State.running

    def kill(self, restart=True):
        print('Killing {}'.format(self))
        '''终止进程服务进程要求的操作'''
        self.state = State.restart if restart else State.zombie

    def create_process(self, user, name):
    '''检查用户权限和生成PID等'''
    print("trying to create the process '{}' for user '{}'".format(name, user))

class WindowServer:
    pass

class NetworkServer:
    pass

class OperatingSystem:
    '''外观'''
    def __init__(self):
        self.fs = FileServer()
        self.ps = ProcessServer()

    def start(self):
        [i.boot() for i in (self.fs, self.ps)]

    def create_file(self, user, name, permissions):
        return self.fs.create_file(user, name, permissions)
    
    def create_process(self, user, name):
        return self.ps.create_process(user, name)

def main():
    os = OperatingSystem()
    os.start()
    os.create_file('foo', 'hello', '-rw-r-r')
    os.create_process('bar', 'ls /tmp')

if __name__ == '__main__':
    main()


# ---享元模式

import random
from enum import Enum

TreeType = Enum('TreeType', 'apple_tree cherry_tree peach_tree')

class Tree:
    pool = dict()

    def __new__(cls, tree_type):
        obj = cls.pool.get(tree_type, None)
        if not obj:
            obj = object.__new__(cls)
            cls.pool[tree_type] = obj
            obj.tree_type = tree_type
        return obj

    def render(self, age, x, y):
        print('render a tree of type {} and age {} at ({}, {})'.format(self.tree_type,
        age, x, y))

def main():
    rnd = random.Random()
    age_min, age_max = 1, 30 # 单位为年
    min_point, max_point = 0, 100
    tree_counter = 0

    for _ in range(10):
        t1 = Tree(TreeType.apple_tree)
        t1.render(rnd.randint(age_min, age_max),
                    rnd.randint(min_point, max_point),
                    rnd.randint(min_point, max_point))
        tree_counter += 1
    
    for _ in range(3):
        t2 = Tree(TreeType.cherry_tree)
        t2.render(rnd.randint(age_min, age_max),
                    rnd.randint(min_point, max_point),
                    rnd.randint(min_point, max_point))
        tree_counter += 1

    for _ in range(5):
        t3 = Tree(TreeType.peach_tree)
        t3.render(rnd.randint(age_min, age_max),
                    rnd.randint(min_point, max_point),
                    rnd.randint(min_point, max_point))
        tree_counter += 1

    print('trees rendered: {}'.format(tree_counter))
    print('trees actually created: {}'.format(len(Tree.pool)))
        
    t4 = Tree(TreeType.cherry_tree)
    t5 = Tree(TreeType.cherry_tree)
    t6 = Tree(TreeType.apple_tree)

    print('{} == {}? {}'.format(id(t4), id(t5), id(t4) == id(t5)))
    print('{} == {}? {}'.format(id(t5), id(t6), id(t5) == id(t6)))

if __name__ == '__main__':
    main()

# ---MVC--- 模式
quotes = ('A man is not complete until he is married. Then he is finished.',
            'As I said before, I never repeat myself.',
            'Behind a successful man is an exhausted woman.',
            'Black holes really suck...', 'Facts are stubborn things.')

class QuoteModel:
    def get_quote(self, n):
        try:
            value = quotes[n]
        except IndexError as err:
            value = 'Not found!'
        return value

class QuoteTerminalView:
    def show(self, quote):
        print('And the quote is: "{}"'.format(quote))
    
    def error(self, msg):
        print('Error: {}'.format(msg))
    
    def select_quote(self):
        return input('Which quote number would you like to see? ')

class QuoteTerminalController:
    def __init__(self):
        self.model = QuoteModel()
        self.view = QuoteTerminalView()
    
    def run(self):
        valid_input = False
        while not valid_input:
            try:
                n = self.view.select_quote()
                n = int(n)
                vaild_input = True
            except ValueError as err:
                self.view.error("Incorrect index '{}'".format(n))
        quote = self.model.get_quote(n)
        self.view.show(quote)

def main():
    controller = QuoteTerminalController()
    while True:
        controller.run()

if __name__ == '__main__':
    main()

# ----代理模式--- #

class SensitiveInfo:
    def __init__(self):
        self.users = ['nick', 'tom', 'ben', 'mike']

    def read(self):
        print('There are {} users: {}'.format(len(self.users), ' '.join(self.users)))

    def add(self, user):
        self.users.append(user)
        print('Added user {}'.format(user))

class Info:
    '''SensitiveInfo的保护代理'''
    def __init__(self):
        self.protected = SensitiveInfo()
        self.secret = '0xdeadbeef'

    def read(self):
        self.protected.read()

    def add(self, user):
        sec = input('what is the secret? ')
        self.protected.add(user) if sec == self.secret else print("That's wrong!")

def main():
    info = Info()

    while True:
        print('1. read list |==| 2. add user |==| 3. quit')
    key = input('choose option: ')
    if key == '1':
        info.read()
    elif key == '2':
        name = input('choose username: ')
        info.add(name)
    elif key == '3':
        exit()
    else:
        print('unknown option: {}'.format(key))

if __name__ == '__main__':
    main()

