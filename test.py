from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import math

# Here need some lists to store the results
trainlist = []
testlist = []

class Domain:
    def __init__(self, _name, _label, _length, _nums, _entropy):
        self.name = _name
        self.label = _label
        self.length = _length
        self.nums = _nums
        self.entropy = _entropy

    def returnInfo(self):
        return [self.length, self.nums, self.entropy]

    def returnLabel(self):
        if self.label == 'notdga':
            return 0
        else:
            return 1

def prepare_4_train(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip() # remove all the spaces
            if line.startswith("#") or line == "":
                # meaning the line is a comment or empty
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]

            # calculate the count of digits in the domain name
            num_count=0
            for i in name:
                if i.isdigit():
                    num_count +=1
            
            # calculate the entropy of this domain name
            count = Counter(name).most_common()
            e=-sum(j/len(name)*(math.log(j/len(name))) for i,j in count)

            # construct list
            trainlist.append(Domain(name, label, len(name), num_count, e))

def prepare_4_test(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                # meaning the line is a comment or empty
                continue
            
            # calculate the count of digits in the domain name
            num_count=0
            for i in line:
                if i.isdigit():
                    num_count +=1
            
            # calculate the entropy of this domain name
            count = Counter(line).most_common()
            e=-sum(j/len(line)*(math.log(j/len(line))) for i,j in count)

            # construct list 
            testlist.append(Domain(line," ", len(line), num_count, e))

if __name__ == '__main__':
    prepare_4_train("train.txt")
    prepare_4_test("test.txt")
    featureMatrix = []
    labellist = []
    for item in trainlist:
        featureMatrix.append(item.returnInfo())
        labellist.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labellist)
    f = open("result.txt", "w")
    for item in testlist:
        res = item.name + ","
        if clf.predict([item.returnInfo()]) == 0:
            res = res + "notdga"
        else:
            res = res + "dga"
        res = res + "\n"
        f.write(res)

    f.close()