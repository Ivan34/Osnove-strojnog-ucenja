numbers = []

while(True):
    print("Unesite broj:")
    try:
        number = input()
        if(number == "Done"):
            break
        number = int(number)
        numbers.append(number)
    except:
        print("Not a number")
        continue

print(len(numbers))
print(sum(numbers) / len(numbers))
print(min(numbers))
print(max(numbers))
numbers.sort()
print(numbers)
