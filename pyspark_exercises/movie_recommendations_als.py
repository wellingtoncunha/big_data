import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating

def loadMovieNames():
    movieNames = {}
    with open("input/moviename.data", encoding="ascii", errors="ignore") as f:
        #contents = f.read().decode("ascii", errors="ignore")
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]
    return movieNames

conf = SparkConf().setMaster("local[*]").setAppName("MoieRecommendationsALS")
sc = SparkContext(conf = conf)
sc.setCheckpointDir("checkpoint")

print("\nLoading movie names...")
nameDict = loadMovieNames()

data = sc.textFile("input/movierating.data")

ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()

# Build the recommendation model using Alternating Least Squares
print("\nTraining recommendation mode...")
rank = 10
# Lowered numIterations to ensure it works on lower-end systems
numIterations = 6
model = ALS.train(ratings, rank, numIterations)

userID = int (sys.argv[1])

print("\nRatings for user ID " + str(userID) + ":")
userRatings = ratings.filter(lambda l: l[0] == userID)
for rating in userRatings.collect():
    print(nameDict[int(rating[1])] + ": " + str(rating[2]))

print("\nTop 10 recommendations:")
recommendations = model.recommendProducts(userID, 10)
for recommendation in recommendations:
    print(nameDict[int(recommendation[1])] + " score " + str(recommendation[2]))