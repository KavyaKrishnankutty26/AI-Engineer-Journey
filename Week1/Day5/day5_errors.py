#try:
    #age = int(input("Enter your age: "))
    #print("Your age is", age)

#except ValueError:
    #print("Please enter a number.")

try:
    number = int(input("Enter a number: "))
    result = 100 / number
    print(result)

except ValueError:
    print("Please enter a valid number.")

except ZeroDivisionError:
    print("You cannot divide by zero.")