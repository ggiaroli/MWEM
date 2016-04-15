import random
import math
#----import numpy - Does not work on IDLE

#To do's:
# 1. Evaluate
# 2. Update

def MWEM(B, Q, T, eps, smart):
    #Initialize real histogram
    print(B)
    rows = max(B[0])
    columns = max(B[1])
    histogram = [[0]*(columns+1) for i in range(rows+1)] #? Correct?
    print(histogram)
    for val in range(len(B[0])):
        histogram[B[0][val]][B[1][val]] += 1
    
    #Initialize synthetic histogram
    nAtt = 2 #Number of attributes (1 for 1D, 2 for 2D)
    A = 0
    n = 0
    if smart:
        #if smart, we spend part of our epsilon budget on making the
        #initial synthetic distribution very more similar to the real one
        weights = [0 for i in range(len(histogram))]
        noise = random.uniform(Laplace(1/(math.e*nAtt*eps)), len(histogram))
        for i in range(len(histogram)):
            weights[i] = max(0.0, histogram[i]+noise-1/(nAtt*eps))
        weights = [(weights[i]*sum(histogram)/sum(weights)) for i in range(len(weights))]
        A0 = weights
        weights = [(0.5*weights[val]+(0.5*sum(histogram)/len(histogram))) for val in range(len(weights))]
        A = weights
        eps = (.5)*eps
    else:
        #Else we simply create a Uniform Distribution
        m = [sum(histogram[i]) for i in range(len(histogram))]
        n = sum(m)
        value = n/(rows*columns)
        A = [[0]*(columns+1) for i in range(rows+1)]
        for i in range(len(histogram)):
            for j in range(len(histogram[i])):
                A[i][j] += value
        
    print("Real: " + str(histogram))
    #print(sum(histogram))
    formattedA = [['%.3f' % elem for elem in A[i]] for i in range(len(A))]
    print("Synthetic: " +str(formattedA))
    print(sum(A))
    print()
    formattedA0 = ['%.3f' % elem for elem in A0]
    print("Synthetic: " +str(formattedA0))
    print(sum(A0))
    return
    measurements = {}

    for i in range(T):
        print("ITERATION #" + str(i))

        #Determine a new query to measure, rejecting prior queries
        qi = ExponentialMechanism(histogram, A, Q, (eps /(2*T)))

        while(qi in measurements):
            print("Into the while ------ ")
            print(qi)
            qi = ExponentialMechanism(histogram, A, Q, eps / (2*T))
            print(qi)
            
        #Measure the query, and add it to our collection of measurements
        print()
        print("INTO Laplace stuff")
        evaluate = Evaluate(Q[qi],histogram)
        lap = Laplace((2*T)/(eps*nAtt))
        measurements[qi] = evaluate + lap
        print("qi: " + str(qi))
        print("Evaluate(Q[qi],histogram): " + str(evaluate))
        print("Laplace: " + str(lap))
        print("Sum: " + str(evaluate + lap))
        print("Measurements: " + str(measurements))
        
        #improve your approximation using poorly fit measurements
        MultiplicativeWeights(A, Q, measurements, histogram)
        print(A)
    return A, histogram

def ExponentialMechanism(B, A, Q, eps):
    #Here we are sampling a query through the exponential mechanism
    #I don't really understand what is happening here!!
    print()
    print("INTO Exponential Mechanism")
    print("len(Q): " + str(len(Q)))
    errors = [0]*len(Q)
    for i in range(len(errors)):
        errors[i] = eps * abs(Evaluate(Q[i], B) - Evaluate(Q[i], A))/2.0
    
    maximum = max(errors)
    print("errors: " + str(errors))
    print("max error: " + str(maximum))
    for i in range(len(errors)):
        errors[i] = math.exp(errors[i] - maximum)
    print()
    print("Errors after subtraction:")
    print(errors)
    
    uniform = sum(errors) * random.random()
    print("uniform: " + str(uniform))
    for i in range(len(errors)):
        uniform -= errors[i]
        print(str(uniform + errors[i]) + " - " + str(errors[i]) + " = " + str(uniform))
        if uniform <= 0.0:
            return i

    return len(errors) - 1


def Laplace(sigma):
    if random.randint(0,1) == 0:
        return sigma * math.log(random.random()) * -1
    else:
        return sigma * math.log(random.random()) * 1


def MultiplicativeWeights(A, Q, measurements, histogram):
    total = sum(A)
    print()
    print("INTO MultiplicativeWeights")
    print("Total: " + str(total))
    for iteration in range(5): #repetitions == 5
        for qi in measurements:

            error = measurements[qi] - Evaluate(Q[qi], A)
            print("Error: " + str(error))
            
            #Update MW!
            for i in range(len(A)):
                #not sure about the following step!! ??????
                A[i] = A[i] * math.exp(histogram[i] * error/(2.0*total))
                #print("A updates..")
                #print(A[i] * math.exp(histogram[i] * error/(2.0*total)))

                
            print("Updated A: " + str(A))

            #Re-normalize!
            count = sum(A)
            print("Count: " + str(count))
            for k in range(len(A)):
                A[k] *= total/count
            print("Normalized A: " + str(A))
            print()
            print("****************************************")
            print()
            if iteration == 4:
                return


def Evaluate(query, collection):
    #We count the number of "objects" from index x
    #to index y specified by the query = {x,y}
    #e.g: collection = [2, 3, 6, 4, 1], query = {(2,1):(3,3)}

    key = list(query)[0]
    startInd = min(key[0], key[1])
    endInd = max(key[0], key[1])
    startInd2 = min(query[key][0], query[key][1])
    endInd2 = max(query[key][0], query[key][1])
    counting = 0
    for i in range(startInd, endInd+1):
        for j in range(startInd2, endInd2+1):
            counting += collection[i][j]
    return counting


def maxError(real, synthetic, Q):
    maxVal = 0
    diff = 0
    for i in range(len(Q)):
        diff = math.abs(Evaluate(Q[i], real) - Evaluate(Q[i], synthetic))
        if diff > maxVal:
            maxVal = diff
    return maxVal  


def meanSqErr(real, synthetic, Q):
    errors = [(Evaluate(Q[i], synthetic) - Evaluate(Q[i], real)) for i in range(len(Q))]
    #-------return (numpy.linalg.norm((errors))**2)/len(errors)

def main():

    #2D Dataset. To do:put code to read data from file!
    B = [[random.randint(0,5) for i in range(6)], [random.randint(0,5) for i in range(6)]] #Dataset

    #bound on the max value possible for the queries, to avoid index out of bound!!
    maxVal1 = max(B[0])
    maxVal2 = max(B[1])
    
    #Queries: count queries for 2D
    Q = [{(random.randint(0,maxVal1),random.randint(0,maxVal1)): (random.randint(0,maxVal2),random.randint(0,maxVal2))} for i in range(7)]
    
    T = 8 #Iterations - HAS to be LESS than the number of queries! 
    eps = 2.0 #Epsilon
    scaleParam = 0 #0 as of now, will see what to do with it - Look at Julia implem.!!
    smart = False #Also look at Julia implem. to see how they use it!!
    
    SintheticData, RealHisto = MWEM(B, Q, T, eps, smart)

    formattedList = ['%.3f' % elem for elem in SintheticData]
    print()
    print("Sinthetic Data: " + str(fomattedList))
    print("Real data histogram: " + str(RealHisto))
    
main()
