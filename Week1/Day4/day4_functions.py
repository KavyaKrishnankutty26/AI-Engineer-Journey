#def greet():
    #print("Hello Kavya!")
#greet()

#def welcome():
    #print ("Welcome to AI Engineering")

#welcome()
#welcome()
#welcome()

#def show_skills():
    #print("Python\n","SQL\n","Git\n","AI\n")
#show_skills()

#def greet(name):
    #print("hello",name)
#greet("Nhythika")   

#def add(a,b):
    #print(a+b)
#add(100,50) 

#def add(a,b):
    #return(a+b)
#result=add(10,20)
#print(result)

#def multiply(a,b):
    #return a*b
#product = multiply(5,4)
#print(product)

#def iseven(number):
    #if number%2 ==0:
       #return True
    #else:
        #return False
#print(iseven(4))
#print(iseven(7))


#def square(number):
    #return number*number
#print(square(5))
#print(square(10))

#def greet(name):
    #print("helo",name)
#greet("kay")
#greet("engineer")

#def multiply(a,b):
    #return a*b
#result=multiply(5,4)
#print(result)

#def greet(name):
    #return name
#call=greet("kavya")
#print("Hello",call)

#def calculate_total(price,quantity):
    #return price*quantity
#total = calculate_total(100,3)
#print (total)

#company={
    #"name":"ABC",
    #"industry":"Technology",
    #"Location":"Dubai"
#}
#print(company["industry"])

#try:
    #age =int(input("enter your age"))
    #print("your age is",age)
#except:
    #print("please enter a valid age")

#def experience():
    #try:
       #number=int(input("Enter your years of experience"))
       #return number
    #except:
       #print("enter valid years of experience")
#exp=experience()
#print("experience is ",exp)


def check_experience():
   

   try:
     experience = int(input("enter your experience"))
     if experience >= 2:
        print("eligible")
     else:
        print("not eligible")
   except:
      print("enter  valid number")
check_experience()