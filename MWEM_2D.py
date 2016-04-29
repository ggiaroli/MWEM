import random
import math
import csv
import numpy #- Does not work on IDLE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


USING_INPUT_DATA = True

def MWEM(B, Q, T, eps, smart, repetitions):
    #Initialize real histogram
    minRow = min(B[0])
    minCol = min(B[1])
    rows = max(B[0]) - min(B[0]) + 1 #age
    columns = max(B[1]) - min(B[1]) + 1 #satisfaction
    
    histogram = matrixCreation(B,rows,columns,minRow,minCol)

    #Initialize synthetic histogram
    nAtt = 2 #Number of attributes (1 for 1D, 2 for 2D)
    A = []
    n = 0
    if smart:
        #if smart, we spend part of our epsilon budget on making the
        #initial synthetic distribution more similar to the real one
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

    measurements = {}

    for i in range(T):
        print("ITERATION #" + str(i))

        #Determine a new query to measure, rejecting prior queries
        qi = ExponentialMechanism(histogram, A, Q, (eps /(2*T)))

        while(qi in measurements):
            qi = ExponentialMechanism(histogram, A, Q, eps / (2*T))

        #Measure the query, and add it to our collection of measurements
        evaluate = Evaluate(Q[qi],histogram)
        lap = Laplace((2*T)/(eps*nAtt))
        measurements[qi] = evaluate + lap

        #improve your approximation using poorly fit measurements
        MultiplicativeWeights(A, Q, measurements, histogram, repetitions)

    return A, histogram

def matrixCreation(B, rows, columns, minRow, minCol):
    histogram = [[0]*(columns) for i in range(rows)] 
    for val in range(len(B[0])):
        histogram[B[0][val] - minRow][B[1][val] - minCol] += 1 
    return histogram

def ExponentialMechanism(B, A, Q, eps):
    #Here we are sampling a query through the exponential mechanism

    errors = [0]*len(Q)
    for i in range(len(errors)):
        errors[i] = eps * abs(Evaluate(Q[i], B) - Evaluate(Q[i], A))/2.0

    #normalize the errors - With these it performs much worse
    #sumErr = sum(errors)
    #newErrors = [errors[i]/sumErr for i in range(len(errors))]
    #errors = newErrors

    maximum = max(errors)
    for i in range(len(errors)):
        errors[i] = math.exp(errors[i] - maximum)
    #print()
    #print("Errors after subtraction:")
    #print(errors)

    uniform = sum(errors) * random.random()
    #print("Sum of errors: " + str(sum(errors)))
    #print("uniform: " + str(uniform))
    for i in range(len(errors)):
        uniform -= errors[i]
     #   print(str(uniform + errors[i]) + " - " + str(errors[i]) + " = " + str(uniform))
        if uniform <= 0.0:
            return i

    #print("Done with Expo Mech.")
    return len(errors) - 1


def Laplace(sigma):
    if random.randint(0,1) == 0:
        return sigma * math.log(random.random()) * -1
    else:
        return sigma * math.log(random.random()) * 1


def MultiplicativeWeights(A, Q, measurements, histogram, repetitions):
    m = [sum(A[i]) for i in range(len(A))]
    total = sum(m)

    for iteration in range(repetitions): #repetitions = 5, testing
        for qi in measurements:

            error = measurements[qi] - Evaluate(Q[qi], A)

            #Update MW!
            query = queryToBinary(Q[qi], len(A[0]), len(A))
            for i in range(len(A)):
                for j in range(len(A[i])):
                    A[i][j] = A[i][j] * math.exp(query[i][j] * error/(2.0*total)) #histogram[i][j]

            #Re-normalize!
            m = [sum(A[i]) for i in range(len(A))]
            count = sum(m)
            for k in range(len(A)):
                for l in range(len(A[k])):
                    A[k][l] *= total/count


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


def queryToBinary(qi, cols, rows):
    binary = [[0]*cols for i in range(rows)] 
    key = list(qi)[0]
    startInd = min(key[0], key[1])
    endInd = max(key[0], key[1])
    startInd2 = min(qi[key][0], qi[key][1])
    endInd2 = max(qi[key][0], qi[key][1])
    for i in range(rows):
        if (i >= startInd) and (i <= endInd):
            for j in range(cols):
                if (j >= startInd2) and (j <= endInd2):
                    binary[i][j] = 1
    return binary


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
    errors = [abs(Evaluate(Q[i], synthetic) - Evaluate(Q[i], real)) for i in range(len(Q))]
    return sum(errors)/len(errors)


def createComplexDataset(size):
#This function generate a random dataset which has a complex distribution
    att1 = [0]*size
    att2 = [0]*size
    for i in range(size):
        ran = random.random() 
        if ran <= .20:
            att1[i] = random.randint(1,700)
            att2[i] = random.randint(0,60)
        elif ran > .2 and ran <=.3:
            att1[i] = random.randint(600,950)
            att2[i] = random.randint(10,60)
        elif ran > .3 and ran <= .45:
            att1[i] = random.randint(222,445)
            att2[i] = random.randint(17,32)
        elif ran > .45 and ran <= .52:
            att1[i] = random.randint(32,420)
            att2[i] = random.randint(41,53)
        elif ran > .52 and ran <= .64:
            att1[i] = random.randint(720,860)
            att2[i] = random.randint(4,9)
        elif ran > .64 and ran <= .8:
            att1[i] = random.randint(0,999)
            att2[i] = random.randint(7,45)
        elif ran > .80 and ran <= .87:
            att1[i] = random.randint(130,520)
            att2[i] = random.randint(51,58)
        elif ran > .87 and ran <= .93:
            att1[i] = random.randint(490,690)
            att2[i] = random.randint(38,60)
        elif ran > .93 and ran <= 1:
            att1[i] = random.randint(660,1000)
            att2[i] = random.randint(2,49)
    
    #Plot
    histogram = plt.figure()
    bins = numpy.linspace(0, 1000+1, 1000+2)
    plt.hist(att1, bins, alpha = 0.75)
    #plt.show
    
    histogram2 = plt.figure()
    bins = numpy.linspace(0, 60+1, 60+2)
    plt.hist(att2, bins, alpha = 0.75)
    #plt.show
    
    #Transfrom into a Matrix
    histo = matrixCreation([att1, att2])
    return histo


def main():

    B = []
    B.append([])
    B.append([])
    if USING_INPUT_DATA == True: #import from file
        with open('Datasets/childMentalHealth_10K.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                    try:
                        B[0].append(int(row[0]))
                        B[1].append(int(row[1]))
                    except ValueError as e:
                        continue
                    except IndexError as e:
                        continue
    else:
        B = [[random.randint(0,100) for i in range(6)], [random.randint(0,100) for i in range(2000)]] #Dataset


    #bound on the max value possible for the queries, to avoid index out of bound!!
    maxVal1 = max(B[0])
    maxVal2 = max(B[1])
    minVal1 = min(B[0])
    minVal2 = min(B[1])

    #Queries: count queries for 2D
    Q = [{(random.randint(0,maxVal1 - minVal1),random.randint(0,maxVal1 - minVal1)): (random.randint(0,maxVal2 - minVal2),random.randint(0,maxVal2 - minVal2))} for i in range(400)]
    
    #For test purposes, we will need to run MWEM more than once and 
    #take an average of the performance metrics, in order to be more reliable
    metrics = [] # to store each running's result
    numRunnings = 1
    for i in range(numRunnings):
        print("------------ MWEM Test Run #" + str(i) + " ------------------")
        print()
        #MWEM Parameters    
        T = 30  #Iterations - HAS to be LESS than the number of queries!
        eps = 3.0  #Epsilon
        scaleParam = 0
        smart = False #Consume some privacy budget for more precise initialization
        repetitions = 20 #Repetitions of Multiplicative Weights mechanism
        
        #Call MWEM
        SintheticData, RealHisto = MWEM(B, Q, T, eps, smart, repetitions)
        
        #Get performance metrics
        maxErr = maxError(RealHisto, SintheticData, Q)
        minErr = minError(RealHisto, SintheticData, Q)
        mse = meanSqErr(RealHisto, SintheticData, Q)
        metrics.append([maxErr, minErr, mse])
        
        print()
        print("Real data histogram: " + str(RealHisto))
        print()
        print("Sinthetic Data: " + str(SintheticData))
        print()
        print("Metrics:")
        print("  - Max Error: " + str(maxErr))
        print("  - Min Error: " + str(minErr))
        print("  - Mean Squared Error: " + str(mse))
        print("  - Mean Error: " + str(meanError(RealHisto, SintheticData, Q)))
        print()
    
    
    #Histogram Plotting
    print("************ REAL DATA *******************")    
    
    H = numpy.array(RealHisto)

    fig = plt.figure(figsize=(4, 4.2))

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
    
    print()
    print("************ SINTHETIC DATA **************")
    
    H2 = numpy.array(SintheticData)

    fig = plt.figure(figsize=(4, 4.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(H2)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    
main()
