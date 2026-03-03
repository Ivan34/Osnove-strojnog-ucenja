def total_euro(hours, pay):
    return hours * pay


print("Unesite radne sate:")
hoursWorked = int(input())
print("Unesite satnicu:")
hourlyPay = float(input())
print("Ukupno", total_euro(hoursWorked, hourlyPay), "eura")
