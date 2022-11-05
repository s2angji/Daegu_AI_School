from audioop import avg
import random

# Number
print(1 + 1) # 2
print(2 - 1) # 1
print(2 * 2) # 4
print(6 / 2) # 3.0
print(pow(4, 3)) # 64
print(4 ** 3) # 64
print(random.random())
print(random.randint(0, 9))

# String
print('Hello')
print("Hello")
print('''
    Hello
    World
''')
print("""
    Hello
    World
""")
print(len('Hello'))
print('Hell World'.replace('Hell', 'Hello'))

# List
member = ['egoing', 'duru', 'taeho']
print(member[0], member[1], member[2]) # egoing duru taeho
print(len(member)) # 3
print(member[random.randint(0, 2)])
print(random.choice(member))
score = [100, 200, 300]
print(sum(score)) # 600