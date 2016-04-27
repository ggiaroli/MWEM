import random
import math
import csv
import matplotlib.pyplot as plt
import numpy #- Does not work on IDLE

#To do's:
# 1. Scale parameter (Look at Julia) - Do we need?
# 2. smart parameter (Look at Julia) - Done!!
# 3. Error function (from Julia) - Done!!
# 4. Read intial dataset from a file - Done!
# 5. Implement for 2D
# 6. Graph

USING_INPUT_DATA=False

def MWEM(B, Q, T, eps, smart, repetitions):
    #Initialize real histogram
    minVal = min(B)
    length = max(B) - minVal + 1
    histogram = [0]*(length)
    for val in range(len(B)):
        histogram[B[val] - minVal] += 1

    #Initialize synthetic histogram
    nAtt = 1 #Number of attributes (1 for 1D, 2 for 2D)
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
        formattedA0 = ['%.3f' % elem for elem in A0]

    else:
        #Else we simply create a Uniform Distribution
        n = sum(histogram)
        A = [n/len(histogram) for i in range(len(histogram))]
        print('Done')

    print("Real: " + str(histogram))
    #print(sum(histogram))
    formattedA = ['%.3f' % elem for elem in A]
    print("Synthetic: " +str(formattedA))
    #print(sum(A))
    #return
    measurements = {}

    for i in range(T):
        print("ITERATION #" + str(i))

        #Determine a new query to measure, rejecting prior queries
        qi = ExponentialMechanism(histogram, A, Q, (eps /(2*T)))

        while(qi in measurements):
            print("INTO THE WHILE ------ ")
            #print(qi)
            qi = ExponentialMechanism(histogram, A, Q, eps / (2*T))
            print("Qi from while loop: " + str(qi))

        #Measure the query, and add it to our collection of measurements
        print()
        print("**********INTO LAPLACE STUFF************")
        evaluate = Evaluate(Q[qi],histogram)
        lap = Laplace((2*T)/(eps*nAtt))
        measurements[qi] = evaluate + lap
        print("Updated measurements: " + str(measurements))

        #improve your approximation using poorly fit measurements
        MultiplicativeWeights(A, Q, measurements, histogram, repetitions)
        #print(A)
    return A, histogram

def ExponentialMechanism(B, A, Q, eps):
    #Here we are sampling a query through the exponential mechanism
    #I don't really understand what is happening here!!
    print()
    print("*******INTO EXPONENTIAL MECHANISM**********")
    #print("len(Q): " + str(len(Q)))
    errors = [0]*len(Q)
    for i in range(len(errors)):
        errors[i] = eps * abs(Evaluate(Q[i], B) - Evaluate(Q[i], A))/2.0

    maximum = max(errors)
    print("errors: " + str(errors))
    print("max error: " + str(maximum))
    for i in range(len(errors)):
        errors[i] = math.exp(errors[i] - maximum)
    #print()
    print("Errors after subtraction: " + str(errors))
    #print(errors)
    
    rNum = random.random()
    uniform = sum(errors) * rNum
    print("Random Number: " + str(rNum))
    print("Uniform: " + str(uniform))
    for i in range(len(errors)):
        uniform -= errors[i]
        #print(str(uniform + errors[i]) + " - " + str(errors[i]) + " = " + str(uniform))
        if uniform <= 0.0:
            print("Returned i!!!")
            return i
    print("Returned len(errors)-1!!!")
    return len(errors) - 1


def Laplace(sigma):
    if random.randint(0,1) == 0:
        return sigma * math.log(random.random()) * -1
    else:
        return sigma * math.log(random.random()) * 1


def MultiplicativeWeights(A, Q, measurements, histogram, repetitions):
    total = sum(A)
    #print()
    print("*****INTO MW****")
    #print("Total: " + str(total))
    for iteration in range(repetitions):
        for qi in measurements:

            error = measurements[qi] - Evaluate(Q[qi], A) #SOMETIMES NEGATIVE, try to understand its effect!!
            print("Get the Error: " + str(error))
            print()
            print("Original A: " + str(A))
            print()
            #Update MW!
            print("A updates..")
            print(str(A[1]) + " becomes --> " + str(A[1] * math.exp(histogram[1] * error/(2.0*total))))
            query = queryToBinary(Q[qi], len(A))
            print(query)
            for i in range(len(A)):     
                #"lenght" of the query
                #key = list(Q[i])[0]
                #length = abs(key - Q[i][key])
                #A[i] = A[i] * math.exp(length * error/(2.0*total))
            
                A[i] = A[i] * math.exp(query[i] * error/(2.0*total))
                
                #!!!!!!!!!
                #histogram[i] doesn't make any fucking sense
                #Plus it is supposed to be a value normalized between 0-1
                #!!!!!!!!!!
                
                #print("A updates..")
                #print(A[i] * math.exp(histogram[i] * error/(2.0*total)))
            

            print("Updated A: " + str(A))
            print()

            #Re-normalize!
            count = sum(A)
            #print("Count: " + str(count))
            for k in range(len(A)):
                A[k] *= total/count
            print("Normalized A: " + str(A))
            print()
            print("********** END OF MW **************")
            print()



def Evaluate(query, collection):
    #We count the number of "objects" from index x
    #to index y specified by the query = {x,y}
    #e.g: collection = [2, 3, 6, 4, 1], query = {2:3}
    # it will sum 6 and 4, returning 10.
    key = list(query)[0]
    startInd = min(key, query[key])
    endInd = max(key, query[key])
    counting = 0
    for i in range(startInd, endInd+1):
        counting += collection[i]
    return counting


def maxError(real, synthetic, Q):
    maxVal = 0
    diff = 0
    for i in range(len(Q)):
        diff = abs(Evaluate(Q[i], real) - Evaluate(Q[i], synthetic))
        if diff > maxVal:
            maxVal = diff
    return maxVal

def meanSqErr(real, synthetic, Q):
    errors = [(Evaluate(Q[i], synthetic) - Evaluate(Q[i], real)) for i in range(len(Q))]
    return (numpy.linalg.norm((errors))**2)/len(errors)
    
def transformForPlotting(Histo, B):
    start = min(B)
    end = max(B)
    newHisto = []
    for i in range(len(Histo)):
        val = math.floor(Histo[i])
        if (Histo[i] - val > .5):
            val = math.ceil(Histo[i])
            
        for j in range(val):
            newHisto.append(start)
        start = start + 1
    return newHisto


def queryToBinary(qi, length):
    binary = [0]*length
    key = list(qi)[0]
    startInd = min(key, qi[key])
    endInd = max(key, qi[key])
    for i in range(length):
        if (i >= startInd) and (i <= endInd):
            binary[i] = 1
    return binary


def main():


    B = []  #1D Dataset

    if USING_INPUT_DATA == True: #import from file
        with open('Datasets/test1.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                    try:
                        B.append(int(row[0]))
                    except ValueError as e:
                        continue
                    except IndexError as e:
                        continue
    else:     #test
        B = [random.randint(0,50) for i in range(500)] #Dataset

    maxVal = max(B)
    minVal = min(B)
    
    #Globals for debugging!!!
    #B = globals()['B']
    #print(globals()['B'])
    
    #Queries: count queries
    Q = [{random.randint(0,maxVal - minVal):random.randint(0,maxVal - minVal)} for i in range(40)] #Queries
    
    #Alternative Queries
    #--Q = [{minVal:maxVal} for i in range(20)]
    Q = [{i:i} for i in range(10)]

    
    T = 10 #Iterations - HAS to be LESS than the number of queries!
    eps = 100.0 #Epsilon
    scaleParam = 0 #0 as of now, will see what to do with it - Look at Julia implem.!!
    smart = False #Also look at Julia implem. to see how they use it!!
    repetitions = 20

    SintheticData, RealHisto = MWEM(B, Q, T, eps, smart, repetitions)

    formattedList = ['%.3f' % elem for elem in SintheticData]
    print()
    print("Sinthetic Data: " + str(formattedList))
    print("Real data histogram: " + str(RealHisto))
    print("MaxError: " + str(maxError(RealHisto,SintheticData, Q)))
    print("MeanSquaredError: " + str(meanSqErr(RealHisto, SintheticData, Q)))
    print("Q: " + str(Q))

    #Histogram Plot
    histogram = plt.figure()
    newHist = transformForPlotting(RealHisto, B)
    bins = numpy.linspace(min(B), max(B)+1, max(B)+2)
    plt.hist(newHist, bins, alpha = 0.75)
    
    newHist2 = transformForPlotting(SintheticData, B)    
    plt.hist(newHist2, bins, color = ['yellow'], alpha = 0.7)
    plt.show()
    
    #Non Histogram plotting
    plt.plot(RealHisto)
    plt.plot(SintheticData, color = 'red')

main()
