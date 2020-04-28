import time
import cv2
from FpMatcher import FpMatcher

#-----------------------------
def genFpImgName(fpNo, imgNo, ext):
    return fpNo + "_" + imgNo + ext
        
#-----------------------------
class FpMatchingEvaluator:
    def __init__(self):
        self.outFileName = "matching_result.txt"
        self.outFile = open(self.outFileName, "w")
        self.outFile.close()
        self.similaritySameFinger = []
        self.similarityDiffFinger = []
        self.falseAccept = [0] * 101
        self.falseReject = [0] * 101
        
    def evaluate(self, dbFileName, fpMatcher):
        print("------------------------------------------------")
        print("Start evaluation")
        print("------------------------------------------------")
        #reset attributes
        self.similaritySameFinger = []
        self.similarityDiffFinger = []
        #open file stream
        inFile = open(dbFileName, "r")
        #get file extension
        ext = inFile.readline().rstrip()
        #get path
        path = inFile.readline().rstrip()
        #get all image names
        line_list = inFile.readlines()
        fpFiles = [x.split() for x in line_list]

        #for all pairs of fingerprint images
        total_time = 0;
        for i in range(len(fpFiles)):
            for j in range(i + 1, len(fpFiles)):
                #read images
                name1 = genFpImgName(fpFiles[i][0], fpFiles[i][1], ext)
                fpImg1 = cv2.imread(path + name1, cv2.IMREAD_GRAYSCALE)
                #fpImg1 = cv2.imread(name1, cv2.IMREAD_GRAYSCALE)
                name2 = genFpImgName(fpFiles[j][0], fpFiles[j][1], ext)
                fpImg2 = cv2.imread(path + name2, cv2.IMREAD_GRAYSCALE)
                #fpImg2 = cv2.imread(name2, cv2.IMREAD_GRAYSCALE)

                #imshow(fpImg1, "Test image 1")
                #imshow(fpImg2, "Test image 2")

                #matching
                start_time = time.time()
                s = fpMatcher.match(fpImg1, fpImg2)
                elapse_time = time.time() - start_time
                total_time += elapse_time
                print(name1, "vs", name2, ":", s)
                self.outFile = open(self.outFileName, "a")
                self.outFile.write(name1 + " vs " + name2 + " : " + str(s) + "\n")
                self.outFile.close()
                print("Time : ", elapse_time, "s. Total time : ", total_time, "s.")
                #save similarity
                if fpFiles[i][0] == fpFiles[j][0]:
                    self.similaritySameFinger.append(s)
                else:
                    self.similarityDiffFinger.append(s)

        #calculate the performace
        self.calculateResults()
                    
    def calculateResults(self):
        print("------------------------------------------------")
        print("Calculate FA & FR at each threshold")
        print("------------------------------------------------")
        self.outFile = open(self.outFileName, "a")
        for th in range(101):
            numFalseAccept = 0
            numFalseReject = 0

            #check genuine attempt vector
            for s in self.similaritySameFinger:
                if s < th:
                    numFalseReject += 1
                    
            #check imposter attempt vector
            for s in self.similarityDiffFinger:
                if s >= th:
                    numFalseAccept += 1

            #compute false acceptance and false rejection rates
            self.falseAccept[th] = numFalseAccept / len(self.similarityDiffFinger) * 100
            self.falseReject[th] = numFalseReject / len(self.similaritySameFinger) * 100

            #display
            print("threshold = ", th, ":", \
                  "false accept = ", self.falseAccept[th], \
                  "false reject = ", self.falseReject[th])
            self.outFile.write("threshold = " + str(th) + ":" + \
                  "false accept = " + str(self.falseAccept[th]) + \
                  "false reject = " + str(self.falseReject[th]) + "\n")

        eer = self.calculateEER()
        print("EER = ", eer)
        self.outFile.write("EER = " + str(eer) + "\n")
        self.outFile.close()

        # plot graphs
        self.plotROC()
        self.plotFAR_FRR()

    def plotROC(self):
        fig = plt.figure()
        plt.plot(self.falseReject, self.falseAccept, color='tab:blue', label='FAR')
        plt.grid()
        plt.xlabel("FRR (%)")
        plt.ylabel("FAR (%)")
        plt.title("ROC")
        plt.show()

    def plotFAR_FRR(self):
        th = np.arange(101)
        fig = plt.figure()
        plt.plot(th, self.falseAccept, color='tab:blue', label='FAR')
        plt.plot(th, self.falseReject, color='tab:orange', label='FRR')
        plt.grid()
        plt.xlabel("Threshold (%)")
        plt.ylabel("FAR & FRR (%)")
        plt.legend()
        plt.show()

    def calculateEER(self): 
        print("------------------------------------------------")
        print("Calculate EER")    
        print("------------------------------------------------")  
        #check if there is a case where FAR equals FRR
        for th in range(101):
            if self.falseReject[th] == self.falseAccept[th]:
                return self.falseReject[th]

        t1 = t2 = 0
        #find EER_low     
        for th in range(101):
            if self.falseReject[th] > self.falseAccept[th]:
                t1 = th
                break

        #find EER_high
        for th in reversed(range(101)):
            if self.falseReject[th] < self.falseAccept[th]:
                t2 = th
                break

        #cal EER
        if (self.falseReject[t1] + self.falseAccept[t1]) < \
           (self.falseReject[t2] + self.falseAccept[t2]):
            eer_low, eer_high = self.falseReject[t1], self.falseAccept[t1]
        else:
            eer_low, eer_high = self.falseAccept[t2], self.falseReject[t2]
        
        return (eer_low + eer_high) / 2


if __name__ == "__main__":
    fpMatcher = FpMatcher()
    fpEvl = FpMatchingEvaluator()
    fpEvl.evaluate("FP (test set)//DB_Info.txt", fpMatcher)
