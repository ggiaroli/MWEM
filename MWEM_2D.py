import random
import math
import csv
import numpy #- Does not work on IDLE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#To do's:
# 1. Evaluate
# 2. Update

USING_INPUT_DATA = True

def MWEM(B, Q, T, eps, smart, repetitions):
    #Initialize real histogram
    rows = max(B[0]) - min(B[0]) + 1 #age
    columns = max(B[1]) - min(B[1]) + 1 #satisfaction

    histogram = [[0]*(columns) for i in range(rows)] #? Correct? #conditional for 0/1
    print("Histogram: " + str(histogram))
    print(len(B[0]))
    print(len(B[1]))
    for val in range(len(B[0])):
        histogram[B[0][val] - min(B[0])][B[1][val] - min(B[1])] += 1

    #Initialize synthetic histogram
    nAtt = 2 #Number of attributes (1 for 1D, 2 for 2D)
    A = []
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
        A = [[0]*(columns) for i in range(rows)]
        for i in range(len(histogram)):
            for j in range(len(histogram[i])):
                A[i][j] += value

    print("Real: " + str(histogram))
    #print(sum(histogram))
    formattedA = [['%.3f' % elem for elem in A[i]] for i in range(len(A))]
    print("Synthetic: " +str(formattedA))
    #formattedA0 = ['%.3f' % elem for elem in A0]
    #print("Synthetic: " +str(formattedA0))
    measurements = {}

    for i in range(T):
        print("ITERATION #" + str(i))

        #Determine a new query to measure, rejecting prior queries
        qi = ExponentialMechanism(histogram, A, Q, (eps /(2*T)))
        print("Measurements: " + str(measurements))
        print()
        while(qi in measurements):
            print("Into the while ------ ")
            print("Before qi: " + str(qi))
            qi = ExponentialMechanism(histogram, A, Q, eps / (2*T))
            print("After qi: " + str(qi))

        #Measure the query, and add it to our collection of measurements
        print("We are free.")
        #print("INTO Laplace stuff")
        evaluate = Evaluate(Q[qi],histogram)
        lap = Laplace((2*T)/(eps*nAtt))
        measurements[qi] = evaluate + lap
        #print("qi: " + str(qi))
        #print("Evaluate(Q[qi],histogram): " + str(evaluate))
        #print("Laplace: " + str(lap))
        #print("Sum: " + str(evaluate + lap))
        #print("Measurements: " + str(measurements))

        #improve your approximation using poorly fit measurements
        MultiplicativeWeights(A, Q, measurements, histogram, repetitions)
        #print(A)
    print("Measurements final: " + str(measurements))
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

    sumErr = sum(errors)
    newErrors = [errors[i]/sumErr for i in range(len(errors))]
    errors = newErrors

    maximum = max(errors)
    print("errors before subtraction: " + str(errors))
    print("max error: " + str(maximum))
    for i in range(len(errors)):
        errors[i] = math.exp(errors[i] - maximum)
    print()
    print("Errors after subtraction:")
    print(errors)

    uniform = sum(errors) * random.random()
    print("Sum of errors: " + str(sum(errors)))
    print("uniform: " + str(uniform))
    for i in range(len(errors)):
        uniform -= errors[i]
        print(str(uniform + errors[i]) + " - " + str(errors[i]) + " = " + str(uniform))
        if uniform <= 0.0:
            return i

    print("Done with Expo Mech.")
    return len(errors) - 1


def Laplace(sigma):
    if random.randint(0,1) == 0:
        return sigma * math.log(random.random()) * -1
    else:
        return sigma * math.log(random.random()) * 1


def MultiplicativeWeights(A, Q, measurements, histogram, repetitions):
    m = [sum(A[i]) for i in range(len(A))]
    total = sum(m)
    print()
    #print("INTO MultiplicativeWeights")
    #print("Total: " + str(total))
    for iteration in range(repetitions): #repetitions = 5, testing
        for qi in measurements:

            error = measurements[qi] - Evaluate(Q[qi], A)
            #print("Error: " + str(error))

            #Update MW!
            for i in range(len(A)):
                #not sure about the following step!! ??????
                for j in range(len(A[i])):
                    A[i][j] = A[i][j] * math.exp(histogram[i][j] * error/(2.0*total)) #histogram[i][j]
                #print("A updates..")
                #print(A[i] * math.exp(histogram[i] * error/(2.0*total)))


            #print("Updated A: " + str(A))

            #Re-normalize!
            m = [sum(A[i]) for i in range(len(A))]
            count = sum(m)
            #print("Count: " + str(count))
            for k in range(len(A)):
                for l in range(len(A[k])):
                    A[k][l] *= total/count
            #print("Normalized A: " + str(A))
            #print()
            #print("****************************************")
            #print()
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
        diff = abs(Evaluate(Q[i], real) - Evaluate(Q[i], synthetic))
        if diff > maxVal:
            maxVal = diff
    return maxVal


def meanSqErr(real, synthetic, Q):
    errors = [(Evaluate(Q[i], synthetic) - Evaluate(Q[i], real)) for i in range(len(Q))]
    return (numpy.linalg.norm((errors))**2)/len(errors)

def minError(real, synthetic, Q):
    minVal = 100000000000
    diff = 0
    for i in range(len(Q)):
        diff = abs(Evaluate(Q[i], real) - Evaluate(Q[i], synthetic))
        if diff < minVal:
            minVal = diff
    return minVal

def meanError(real, synthetic, Q):
    errors = [(Evaluate(Q[i], synthetic) - Evaluate(Q[i], real)) for i in range(len(Q))]
    return sum(errors)/len(errors)

def main():

    #2D Dataset. To do:put code to read data from file!
    #1D Dataset
    B = []
    B.append([])
    B.append([])
    if USING_INPUT_DATA == True: #import from file
        with open('Datasets/diabeticAge-10*3.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                    try:
                        B[0].append(int(row[0]))
                        B[1].append(int(row[1]))
                    except ValueError as e:
                        continue
                    except IndexError as e:
                        continue
    else:     #test
        B = [[random.randint(0,5) for i in range(6)], [random.randint(0,5) for i in range(6)]] #Dataset
    print("B: " + str(B))

    TEST = False

    if TEST:
        print("Done")
    else:
        #bound on the max value possible for the queries, to avoid index out of bound!!
        maxVal1 = max(B[0])
        maxVal2 = max(B[1])
        minVal1 = min(B[0])
        minVal2 = min(B[1])

        print("Max Val 1: " + str(maxVal1))
        print("Max Val 2: " + str(maxVal2))

        #Queries: count queries for 2D
        Q = [{(random.randint(0,maxVal1 - minVal1),random.randint(0,maxVal1 - minVal1)): (random.randint(0,maxVal2 - minVal2),random.randint(0,maxVal2 - minVal2))} for i in range(200)]
        print()
        print("Q: " + str(Q))
        T = 30 #Iterations - HAS to be LESS than the number of queries!
        eps = 6.0 #Epsilon
        scaleParam = 0 #0 as of now, will see what to do with it - Look at Julia implem.!!
        smart = False #Also look at Julia implem. to see how they use it!!
        repetitions = 200

        SintheticData, RealHisto = MWEM(B, Q, T, eps, smart, repetitions)

        #formattedList = ['%.3f' % elem for elem in SintheticData]
        print()
        print("Sinthetic Data: " + str(SintheticData))
        print("Real data histogram: " + str(RealHisto))
        print("Max Error: " + str(maxError(RealHisto, SintheticData, Q)))
        print("Min Error: " + str(minError(RealHisto, SintheticData, Q)))
        print("Mean Squared Error: " + str(meanSqErr(RealHisto, SintheticData, Q)))
        print("Mean Error: " + str(meanError(RealHisto, SintheticData, Q)))

        H = numpy.array(SintheticData)

        fig = plt.figure(figsize=(12, 3.2))

        ax = fig.add_subplot(111)
        ax.set_title('colorMap')
        plt.imshow(H)
        ax.set_aspect('equal')

        cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.patch.set_alpha(0)
        cax.set_frame_on(False)
        plt.colorbar(orientation='vertical')
        plt.show()


main()
