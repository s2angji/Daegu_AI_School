members = ['egoing', 'duru']
for member in members:
    print(member)
for i in range(0, len(members)):
    print(members[i])

gugu = [[x * y for x in range(1, 10)] for y in range(2, 10)] # 내포
for x in gugu:
    print(x)

members = [
    ['egoing', 'seoul', 'programmer'],
    ['duru', 'daegu', 'dba']
]
print(members[0][0])
for member in members:
    print(member)

egoing = ['egoing', 'seoul', 'programmer']
egoing = {'name': 'egoing', 'city': 'seoul', 'job': 'programmer'} # 사전형, dictionary
print(egoing['city']) # seoul
for name in egoing:
    print(egoing[name])

members = [
    {'name': 'egoing', 'city': 'seoul', 'job': 'programmer'},
    {'name': 'duru', 'city': 'daegu', 'job': 'dba'}
]

for member in members:
    for item in member:
        print(f'{item} = {member[item]}')