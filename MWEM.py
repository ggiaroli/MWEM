import random
import math
import csv
import matplotlib.pyplot as plt
#----import numpy - Does not work on IDLE

#To do's:
# 1. Scale parameter (Look at Julia) - Do we need?
# 2. smart parameter (Look at Julia) - Done!!
# 3. Error function (from Julia) - Done!!
# 4. Read intial dataset from a file
# 5. Implement for 2D
# 6. Graph

USING_INPUT_DATA=True

def MWEM(B, Q, T, eps, smart):
    #Initialize real histogram
    length = max(B)
    histogram = [0]*(length + 1)
    for val in range(len(B)):
        histogram[B[val]] += 1

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
        print()
        formattedA0 = ['%.3f' % elem for elem in A0]
        print("Synthetic: " +str(formattedA0))
        print(sum(A0))
    else:
        #Else we simply create a Uniform Distribution
        n = sum(histogram)
        A = [n/len(histogram) for i in range(len(histogram))]

    #print("Real: " + str(histogram))
    #print(sum(histogram))
    #formattedA = ['%.3f' % elem for elem in A]
    #print("Synthetic: " +str(formattedA))
    #print(sum(A))
    #return
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
        #print("qi: " + str(qi))
        #print("Evaluate(Q[qi],histogram): " + str(evaluate))
        #print("Laplace: " + str(lap))
        #print("Sum: " + str(evaluate + lap))
        #print("Measurements: " + str(measurements))

        #improve your approximation using poorly fit measurements
        MultiplicativeWeights(A, Q, measurements, histogram)
        #print(A)
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
    #print("errors: " + str(errors))
    #print("max error: " + str(maximum))
    for i in range(len(errors)):
        errors[i] = math.exp(errors[i] - maximum)
    print()
    #print("Errors after subtraction:")
    #print(errors)

    uniform = sum(errors) * random.random()
    #print("uniform: " + str(uniform))
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

            error = measurements[qi] - Evaluate(Q[qi], A) #SOMETIMES NEGATIVE, try to understand its effect!!
            print("Error: " + str(error))

            #Update MW!
            for i in range(len(A)):
                #not sure about the following step!! ??????
                A[i] = A[i] * math.exp(histogram[i] * error/(2.0*total))
                #print("A updates..")
                #print(A[i] * math.exp(histogram[i] * error/(2.0*total)))


            #print("Updated A: " + str(A))

            #Re-normalize!
            count = sum(A)
            print("Count: " + str(count))
            for k in range(len(A)):
                A[k] *= total/count
            #print("Normalized A: " + str(A))
            #print()
            #print("****************************************")
            #print()
            if iteration == 4:
                return


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
    #-------return (numpy.linalg.norm((errors))**2)/len(errors)

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
        B = [random.randint(1,15) for i in range(30)] #Dataset

    maxVal = max(B)

    #Queries: count queries
    Q = [{random.randint(1,maxVal):random.randint(1,maxVal)} for i in range(12)] #Queries

    T = 8 #Iterations - HAS to be LESS than the number of queries!
    eps = 3.0 #Epsilon
    scaleParam = 0 #0 as of now, will see what to do with it - Look at Julia implem.!!
    smart = False #Also look at Julia implem. to see how they use it!!

    SintheticData, RealHisto = MWEM(B, Q, T, eps, smart)

    formattedList = ['%.3f' % elem for elem in SintheticData]
    print()
    print("Sinthetic Data: " + str(formattedList))
    print("Real data histogram: " + str(RealHisto))
    print("Error: " + str(maxError(RealHisto,SintheticData, Q)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    numBins = 5
    ax.hist(RealHisto,numBins,color=['green'],alpha=0.8)

    bx = fig.add_subplot(111)
    bx.hist(SintheticData, numBins,color=['red'],alpha=0.8)

    plt.show()


main()
