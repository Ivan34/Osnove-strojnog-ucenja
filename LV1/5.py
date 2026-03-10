totalHam = 0
totalSpam = 0
spamWords = 0
hamWords = 0
endWithExclCount = 0

try:
    fhand = open("SMSSpamCollection.txt")
    for line in fhand:
        line = line.rstrip()

        parts = line.split(maxsplit=1)
        if(len(parts) == 2):
            tag = parts[0]
            message = parts[1]
            words = message.split()
            if(tag == "ham"):
                totalHam += 1
                hamWords += (len(words))
            elif(tag == "spam"):
                totalSpam += 1
                spamWords += (len(words))
            if(message.endswith("!")):
                endWithExclCount+=1

    print("Prosjecan broj ham rijeci: ")

except FileNotFoundError:
    print("File not found")