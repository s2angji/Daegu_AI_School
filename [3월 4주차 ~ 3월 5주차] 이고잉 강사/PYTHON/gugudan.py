import random
def quiz(x = random.randint(1, 9), y = random.randint(2, 9)):
    gugudan = [[x * y for x in range(0, 10)] for y in range(0, 10)]
    if int(input(f'{x}*{y} = ')) == gugudan[x][y]:
        print('정답입니다')
    else:
        print('틀렸습니다.')

quiz()