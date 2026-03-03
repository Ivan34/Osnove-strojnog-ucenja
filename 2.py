
try:
    print("Unesite broj izmedu 0.0 i 1.0:")
    grade = float(input())
    if(grade > 1.0 or grade < 0.0):
        print("Pogresan interval broja")
    elif(grade >= 0.9):
        print("A")
    elif(grade >= 0.8):
        print("B")
    elif(grade >= 0.7):
        print("C")
    elif(grade >= 0.6):
        print("D")
    elif(grade < 0.6):
        print("F")
except:
    print("Nije unesen broj")
