def foo():
    a = 1
    def fa():
        print(a)
        a+=1
        print(a)
    
    def fb(a):
        def apply():
            print(a)
        return apply
    
    fc = lambda: print(a)
    return fa, fb(a), fc

fa, fb, fc = foo()

fa()
fb()
fc()
