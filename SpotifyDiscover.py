# An MLlib & PySpark implementation of audio recommendation system using a collaborative filtering algorithm
# Spotify's "Discover Weekly" built with Spark
# https://towardsdatascience.com/building-spotifys-discover-weekly-with-spark-4370d5d0df2f

#content filtering: using known information about the products and users to make recommendations
#profiles created based on products and users
#collaborative filtering: uses previous users' input/behaviour to make future recommendations
#ignore any a priori user or object information
#use the ratings of similar users to predict the rating

import findspark
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib import recommendation
from pyspark.mllib.recommendation import *

#initialize Spark in VM

findspark.init('/usr/local/bin/spark-1.3.1-bin-hadoop2.6/')
try:
    sc=SparkContext()
except:
    None 

#define variables

rawUserArtistData = sc.textFile("vagrant/user_artist_data.txt")
rawArtistData = sc.textFile("vagrant/artist_data.txt")
rawArtistAlias = sc.textFile("vagrant/artist_alias.txt")

#preprocess data

def pairsplit(singlePair):
	splitPair = singlePair.rsplit('\t')
	if len(splitPair) != 2:
		return []
	else:
		try:
			return [(int(splitPair[0]), splitPair[1])]
		except:
			return []

artistByID = dict(rawArtistData.flatMap(lambda x: pairsplit(x)).collect())

def aliaslookup(alias):
	splitPair = alias.rsplit('\t')
	if len(splitPair) != 2:
		return []
	else: 
		try:
			return [(int(splitPair[0]), int(splitPair[1]))]
		except:
			return []

	artistAlias = rawArtistAlias.flatMap(lambda x:
		aliaslookup(x)).collectAsMap()

def ratinglookup(x):
	userID, artistID, count = map(lambda line: int(line), x.split())
	finalArtistID = bArtistAlias.value.get(artistID)
	if finalArtistID is None:
		finalArtistID = artistID
	return Rating(userID, finalArtistID, count)

trainData = rawUserArtistData.map(lambda x: ratinglookup(x))
tranData.cache()

bArtistAlias = sc.broadcast(artistAlias)

#build model

model = ALS.trainImplicit(trainData, 10, 5)

#test artist

spotcheckingID = 2093760
bArtistByID = sc.braodcast(artistByID)

rawArtistForUser = (trainData
					.filter(lambda x: x.user == spotcheckingID)
					.map(lambda x: bArtistByID.value.get(x.product))
					.collect())
print(rawArtistForUser)

#output recommendations

recommendations = map(lambda x: artistByID.get(x.product),
					model.call("recommendProducts", spotcheckingID, 10))
print(recommendations)

