wordsdic = {}
uniqueWords = 0

try:
    fhand = open("song.txt")
    for line in fhand:
        line = line.rstrip()
        words = line.split()

        for word in words:
           word = word.lower().rstrip("?!.,")
           if word in wordsdic:
               wordsdic[word] = wordsdic[word] + 1
           else:
               wordsdic[word] = 1
    print(wordsdic)
    for word in wordsdic:
        if(wordsdic[word] == 1):
            print(word)
            uniqueWords += 1
    print("Number of unique words: ", uniqueWords)

except FileNotFoundError:
    print("File not fount")