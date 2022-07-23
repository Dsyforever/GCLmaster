import time


class logs:
    def __init__(self):
        self.time_begin=0
        self.time_end=0
        self.train_lossdata=[]
        self.test_lossdata=[]




t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(t)
x=[1,2,3,4,5]
y=x[-1]
print(y)