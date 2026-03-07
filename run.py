import math
n=1
while True:
    num = n*(10**len(str(n))+1)
    if math.sqrt(num)==int(math.sqrt(num)):
        print(len(str(n)),math.sqrt(num),n,num)
    n+=1

