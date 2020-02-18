import json

def main():
    fileName = 'vertical'
    begining = 0
    ending = 300
    toCorrect = {}
    inverseDict = {}
    with open(fileName + '.json', 'r') as fp:
        toCorrect = json.load(fp)
   
    for key,val in toCorrect.items():
        key = int(key)
        inverseDict[val] = key


    previous = 0
    for i in range(ending):
        if i in inverseDict.keys():
            previous = inverseDict[i]
        else:
            inverseDict[i] = previous


    with open(fileName + 'Corrected' + '.json', 'w') as fp:
        json.dump(inverseDict, fp)


if __name__ == '__main__':
    main()