# encoding: utf-8
from __future__ import division, print_function
from SWAPR3 import *
from SWAPR3weights import *
from itertools import groupby
from numpy import median, mean

class calibAlg:
	def __init__(self,name,sumFunc,weightFunc=weightBIBI,longName = '',offsetStyle = None):
		self.sumFunc = sumFunc
		self.weightFunc = weightFunc
		self.offsetStyle = offsetStyle
		self.name = name
		if longName == '':
			self.longName = name
def assignCalibrationGrades(db,labNumber, nCalibration=3, term='F2014'):
    'Give each student a calibration grade (out of 100) for each labNumber. Each itemIndex has a maximum weight of 1; there are R weighted (i.e. graded) items per rubric.'
    # First, get R
    R = db.getNgradedItems(labNumber)
    db.cursor.execute('''SELECT wID, weightType, sum(weight)
        FROM weights
        WHERE
            labNumber = ?
            AND nCalibration = ?
            AND weight IS NOT NULL
        GROUP BY wID, weightType
        ORDER BY wID, weightType
        ''',[labNumber,nCalibration])
    data = [ [str(entry[0]), str(entry[1]), float(entry[2])] for entry in db.cursor.fetchall() ]
    for wID, wIDgroup in groupby(data, key = lambda x: x[0]):
        for entry in wIDgroup:
            weightType = entry[1]
            rawScore = entry[2]
            grade = 100*rawScore/R
            db.cursor.execute('''INSERT INTO calibrationGrades
                VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',[datetime.now(), term, labNumber, wID, nCalibration, rawScore, grade, weightType])
    db.conn.commit()
def getNstudents(db,URL):
	db.cursor.execute("SELECT COUNT(DISTINCT wID) FROM responses WHERE URL = ? AND response is not NULL",[URL])
	return(int(db.cursor.fetchone()[0]))

def ncr(n, r):
	r = min(r, n-r)
	if r == 0: return 1
	numer = reduce(op.mul, xrange(n, n-r, -1))
	denom = reduce(op.mul, xrange(1, r+1))
	return numer//denom


def semFinite(data,N):
	'Standard error of the mean with finite population correction'
	# print(N)
	# print(len(data))
	if len(data) < 0.05*N:
		return sem(data)
	else:
		return sem(data)*((N-len(data))/(N-1))**0.5

def weightedSumMedianFallback(weights,scores,offsets=None,maxScores=None):
	# Computes a weighted sum (weights[i][j]*scores[i][j])/sum(weights[i][j]) for each j; if all weights[i][j] are 0 for a given j, then finalGradeVector[j] is the median of all scores[i][j]
	R = len(weights[0]) # Number of graded rubric items
	N = len(weights)    # Number of peer grades
	numerators = [0]*R
	denominators = [0]*R
	for j in range(R):
		for i in range(N):
			if len(weights[i]) == R and len(scores[i]) == len(weights[i]):
				numerators[j] += weights[i][j]*scores[i][j]
				denominators[j] += weights[i][j]
		if denominators[j] == 0:
			# raise Exception('All peers have weight = 0')
			# print('All peers have weight=0')
			numerators[j] = median([score[j] for score in scores])
			denominators[j] = 1
	finalGradeVector = [numerators[i]/denominators[i] for i in range(R)]
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def weightedSumMeanFallback(weights,scores,offsets=None,maxScores=None):
	# Computes a weighted sum (weights[i][j]*scores[i][j])/sum(weights[i][j]) for each j; if all weights[i][j] are 0 for a given j, then finalGradeVector[j] is the mean of all scores[i][j]
	R = len(weights[0]) # Number of graded rubric items
	N = len(weights)    # Number of peer grades
	numerators = [0]*R
	denominators = [0]*R
	for j in range(R):
		for i in range(N):
			if len(weights[i]) == R and len(scores[i]) == len(weights[i]):
				numerators[j] += weights[i][j]*scores[i][j]
				denominators[j] += weights[i][j]
		if denominators[j] == 0:
			# print('All peers have weight=0')
			numerators[j] = mean([score[j] for score in scores])
			denominators[j] = 1
	finalGradeVector = [numerators[i]/denominators[i] for i in range(R)]
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def sumWinnersMedian(weights,scores,offsets=None,maxScores=None):
	# For each j, picks out the highest weights[i][j] and assigns the corresponding score to finalGradeVector[j]. In the event of a tie, assign the median score. The median of an even number of scores is the mean of the two central scores.
	R = len(scores[0]) # Number of graded rubric items
	finalGradeVector = R*[0]
	for j in range(R):
		jWeights = [weight[j] for weight in weights if len(weight) == R]
		# Get the indices of the graders with the highest weights for item j
		winners = [index for index, weight in enumerate(jWeights) if weight == max(jWeights)]
		finalGradeVector[j] = median([scores[index][j] for index in winners if len(scores[index]) == R])
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def sumWinnersMean(weights,scores,offsets=None,maxScores=None):
	# For each j, picks out the highest weights[i][j] and assigns the corresponding score to finalGradeVector[j]. In the event of a tie, assign the mean score.
	R = len(scores[0]) # Number of graded rubric items
	finalGradeVector = R*[0]
	for j in range(R):
		# Get the indices of the graders with the highest weights for item j
		jWeights = [weight[j] for weight in weights if len(weight) == R]
		winners = [index for index, weight in enumerate(jWeights) if weight == max(jWeights)]
		finalGradeVector[j] = mean([scores[index][j] for index in winners if len(scores[index]) == R])
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def sumNoZeroesMedian(weights,scores,offsets=None,maxScores=None):
	# Throw out all zero-weighted scores and take the median of the remaining scores. Otherwise take the median of all scores.
	R = len(scores[0]) # Number of graded rubric items
	finalGradeVector = R*[0]
	for j in range(R):
		jWeights = [weight[j] for weight in weights if len(weight) == R]
		nonZeroes = [index for index, weight in enumerate(jWeights) if weight != 0]
		if len(nonZeroes) > 0:
			finalGradeVector[j] = median([scores[index][j] for index in nonZeroes if len(scores[index]) == R])
		else:
			finalGradeVector = [ median([score[j] for score in scores if len(score) == R]) for j in range(R) ]
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def sumMedian(weights,scores,offsets=None,maxScores=None):
	# For each j, return the median of all scores[i][j]
	R = len(scores[0])
	finalGradeVector = [ median([score[j] for score in scores if len(score) == R]) for j in range(R) ]
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def sumMean(weights,scores,offsets=None,maxScores=None):
	# For each j, return the mean of all scores[i][j]
	R = len(scores[0])
	finalGradeVector = [ mean([score[j] for score in scores if len (score) == R]) for j in range(R)  ]
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def weightedSumOffset(weights,scores,offsets=None,maxScores=None):
	# For each j, return the weighted mean of all scores[i][j] with a median fallback
	R = len(scores[0])
	N = len(scores)
	numerators = [0]*R
	denominators = [0]*R
	for j in range(R):
		numerators[j] = sum([abs(weights[i][j])*(scores[i][j]-offsets[i][j]) for i in range(N)])
		denominators[j] = sum([abs(weights[i][j]) for i in range(N)])
		if denominators[j] == 0:
			# print('All peers have weight=0')
			numerators[j] = median([scores[i][j] for i in range(N)])
			denominators[j] = 1
	finalGradeVector = [min([ max([numerators[j]/denominators[j], 0]), maxScores[j] ]) for j in range(R)]
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def weightedSumOffsetMeanFallback(weights,scores,offsets=None,maxScores=None):
	# For each j, return the weighted mean of all scores[i][j] with a median fallback
	R = len(scores[0])
	N = len(scores)
	numerators = [0]*R
	denominators = [0]*R
	for j in range(R):
		numerators[j] = sum([weights[i][j]*(scores[i][j]-offsets[i][j]) for i in range(N)])
		denominators[j] = sum([weights[i][j] for i in range(N)])
		if denominators[j] == 0:
			# print('All peers have weight=0')
			numerators[j] = mean([scores[i][j] for i in range(N)])
			denominators[j] = 1
	finalGradeVector = [min([ max([numerators[j]/denominators[j], 0]), maxScores[j] ]) for j in range(R)]
	finalGrade = sum(finalGradeVector)
	return finalGrade, finalGradeVector

def getWeightsScores(db,URL,weightFunc=weightBIBI,offsetStyle=None):
	# get the labNumber for that URL
	db.cursor.execute("SELECT labNumber FROM responses WHERE URL = ?",[URL])
	labNumber = int(db.cursor.fetchone()[0])

	# get the number of graded items (R) for that labNumber
	db.cursor.execute("SELECT count(itemIndex) FROM rubrics WHERE labNumber = ? AND graded",[labNumber])
	R = int(db.cursor.fetchone()[0])

	offsets = []
	if offsetStyle != None:
		db.cursor.execute("SELECT responses.URL, responses.wID, responses.itemIndex, weight, ratingKeys.score FROM responses, weights, ratingKeys, rubrics WHERE responses.URL = ? AND weights.labNumber = responses.labNumber AND ratingKeys.labNumber = responses.labNumber AND rubrics.labNumber = responses.labNumber AND responses.wID = weights.wID AND weights.itemIndex = responses.itemIndex AND rubrics.itemIndex = responses.itemIndex AND ratingKeys.itemIndex = responses.itemIndex AND responses.response = ratingKeys.response AND weights.weightType = ? AND rubrics.graded ORDER BY responses.URL, responses.wID, responses.itemIndex",[URL,offsetStyle])
		data = db.cursor.fetchall()
		for wID, wIDoffsets in groupby(list(data), key=lambda x: str(x[1])):
			thisOffset = []
			for entry in list(wIDoffsets):
				# append every weight value
				thisOffset.append(float(entry[3]))
			if len(thisOffset) == R:
				offsets.append(thisOffset)    
	if weightFunc == None:
		db.cursor.execute("SELECT responses.URL, responses.wID, responses.itemIndex, 1, ratingKeys.score FROM responses, ratingKeys, rubrics WHERE responses.URL = ? AND ratingKeys.labNumber = responses.labNumber AND rubrics.labNumber = responses.labNumber AND rubrics.itemIndex = responses.itemIndex AND ratingKeys.itemIndex = responses.itemIndex AND responses.response = ratingKeys.response AND rubrics.graded ORDER BY responses.URL, responses.wID, responses.itemIndex",[URL])
	else:
		db.cursor.execute("SELECT responses.URL, responses.wID, responses.itemIndex, weight, ratingKeys.score FROM responses, weights, ratingKeys, rubrics WHERE responses.URL = ? AND weights.labNumber = responses.labNumber AND ratingKeys.labNumber = responses.labNumber AND rubrics.labNumber = responses.labNumber AND responses.wID = weights.wID AND weights.itemIndex = responses.itemIndex AND rubrics.itemIndex = responses.itemIndex AND ratingKeys.itemIndex = responses.itemIndex AND responses.response = ratingKeys.response AND weights.weightType = ? AND rubrics.graded ORDER BY responses.URL, responses.wID, responses.itemIndex",[URL,weightFunc.__name__])
	data = db.cursor.fetchall()
	# sort by wID
	weights = []
	scores = []
	for wID, wIDweights in groupby(list(data), key=lambda x: str(x[1])):
		thisWeight = []
		thisScore = []

		for entry in list(wIDweights):
			# append every weight value
			thisWeight.append(float(entry[3]))
			thisScore.append(float(entry[4]))
		if len(thisWeight) == len(thisScore) and len(thisWeight) == R:
			weights.append(thisWeight)
			scores.append(thisScore)

	return weights, scores, offsets, itemIndices

def randomCombinations(iterable, N, n):
	"Random selection of N combinations from itertools.combinations(iterable, n)"
	# Return all combinations if N >= (total number of combinations)
	# The number of combinations can get very large!
	# pool = tuple(combinations(iterable, n))
	# print(len(pool))
	# Ncombos = len(pool)
	# indices = sorted(random.sample(xrange(Ncombos), min(N,Ncombos)))
	# return list(pool[i] for i in indices)
	combos = []
	while len(combos) < min(N,ncr(len(iterable),n)):
		# print('Generating combo '+str(len(combos) + 1)+'...')
		indices = sorted(random.sample(xrange(len(iterable)), n))
		combo = [iterable[index] for index in indices]
		if combo not in combos: # This could take a very long time for large N, but hopefully not as long as generating tuple(combinations(iterable, n))
			combos.append(combo)
	return combos

def getExpertGrade(db,expertURL):
	db.cursor.execute('''SELECT e.itemIndex, e.response, k.score
		FROM experts e, rubrics r, ratingKeys k 
		WHERE 
			URL = ? 
			AND e.labNumber = r.labNumber 
			AND e.labNumber = k.labNumber 
			AND e.itemIndex = r.itemIndex 
			AND e.itemIndex = k.itemIndex 
			AND e.response = k.response 
			AND graded 
		ORDER BY e.itemIndex''',[expertURL])
	data = db.cursor.fetchall()
	expertGradeVector = [float(entry[2]) for entry in data]
	expertGrade = sum(expertGradeVector)
	return expertGrade, expertGradeVector

def getCalibratedGrade(db,expertURL,alg):
	weights, scores, offsets = getWeightsScores(db,expertURL,alg.weightFunc,alg.offsetStyle)
	return alg.sumFunc(weights,scores)

def squareError(a,b):
	if len(a) == len(b):
		sqError = sum([ (a[i]-b[i])**2 for i in range(len(a))])
		return sqError
	else:
		print('Two lists are not the same length')

def absError(a,b):
	if len(a) == len(b):
		aError = sum([ abs(a[i]-b[i]) for i in range(len(a))])
		return aError
	else:
		print('Two lists are not the same length')

def generateSampleGrades(db,expertURL,n,alg,matchmaking='random',nCalibration=3):
	db.cursor.execute("SELECT DISTINCT labNumber FROM experts WHERE URL = ?",[expertURL])
	labNumber = int(db.cursor.fetchone()[0])
	maxScore, maxScoreVector = getMaxScore(db,labNumber)
	if alg.offsetStyle == None:
		db.cursor.execute('''SELECT r.wID, r.itemIndex, k.score, w.weight, 0
			FROM responses r, rubrics rub, ratingKeys k, weights w
			WHERE 
				--only one URL
				r.URL = ? 
				--match wID
				AND r.wID = w.wID
				--match itemIndex
				AND r.itemIndex = rub.itemIndex 
				AND r.itemIndex = k.itemIndex 
				AND r.itemIndex = w.itemIndex 
				--match labNumber
				AND r.labNumber = rub.labNumber 
				AND r.labNumber = k.labNumber 
				AND r.labNumber = w.labNumber
				--only graded items
				AND rub.graded
				--match score to student response
				AND r.response = k.response 
				--get the right weight
				AND w.weightType = ?
				AND w.nCalibration = ?
				ORDER BY r.wID, r.itemIndex
				''',[expertURL,alg.weightFunc.__name__,nCalibration])

	else:
		db.cursor.execute('''SELECT r.wID, r.itemIndex, k.score, w.weight, off.weight
			FROM responses r, rubrics rub, ratingKeys k, weights w, weights off 
			WHERE 
				--only one URL
				r.URL = ? 
				--match wIDs
				AND r.wID = w.wID
				AND r.wID = off.wID
				--match itemIndex
				AND r.itemIndex = rub.itemIndex 
				AND r.itemIndex = k.itemIndex 
				AND r.itemIndex = w.itemIndex 
				AND r.itemIndex = off.itemIndex 
				--match labNumber
				AND r.labNumber = rub.labNumber 
				AND r.labNumber = k.labNumber 
				AND r.labNumber = w.labNumber
				AND r.labNumber = off.labNumber
				--only graded items
				AND rub.graded
				--match score to student response
				AND r.response = k.response 
				--get the right weight
				AND w.weightType = ?
				AND w.nCalibration = ?
				--get the right offset
				AND off.weightType = ?
				AND off.nCalibration = ?
				ORDER BY r.wID, r.itemIndex
				''',[expertURL,alg.weightFunc.__name__ if alg.weightFunc != None else 'weightBIBI',nCalibration,alg.offsetStyle,nCalibration])
	data = db.cursor.fetchall()
	studentScores = {}
	for wID, wIDgroup in groupby(data, key = lambda x: x[0]):
		if wID == 'menriquez3@gatech':
			print('Found menriquez3@gatech')
		wIDgroup = list(wIDgroup)
		thisItemIndices = [int(entry[1]) for entry in wIDgroup]
		thisScores = [float(entry[2]) for entry in wIDgroup]
		thisWeights = [float(entry[3]) for entry in wIDgroup]
		thisOffsets = [float(entry[4]) for entry in wIDgroup]
		studentScores.update({wID:{'itemIndices':thisItemIndices,'scores':thisScores,'weights':thisWeights,'offsets':thisOffsets}})
	# Now get the peerGroups we're going to use, and split their wIDs up into lists of the appropriate size
	db.cursor.execute('''SELECT g.peerGroup, g.wID from peerGroups p, groupMembers g
		WHERE
			--match peerGroup
			p.peerGroup = g.peerGroup
			--get the right nGroups
			AND p.nGroup = ?
			ORDER BY g.peerGroup
			''',[n])
	data = db.cursor.fetchall()
	peerGroups = []
	for peerGroupID, group in groupby(data, key = lambda x: x[0]):
		peerGroups.append([peerGroupID,[str(entry[1]) for entry in group]])

	# MAIN SIMULATION LOOP
	db.cursor.execute("INSERT INTO simulations VALUES (NULL,?,?,?,?,?)",[expertURL,len(peerGroups),n,alg.name,nCalibration])
	expertFinalGrade, expertScore = getExpertGrade(db,expertURL)
	for peerGroup in peerGroups:
		peerGroupID = peerGroup[0]
		wIDs = peerGroup[1]
		# The pureOff style requires every score to be paired with a constant weight; by convention, weight = 1
		weights = [studentScores[wID]['weights'] for wID in wIDs] if alg.name != 'pureOff' else [ [1 for entry in studentScores[wID]['weights'] ] for wID in wIDs]
		scores = [studentScores[wID]['scores'] for wID in wIDs]
		# When drawing student responses from among the groupMembers table, we assert that every student has completed every graded item for every expert video. If this is not the case, then the first student in the peerGroup might not have the same itemIndices as everyone else, and the next line will cause bad behavior
		itemIndices = studentScores[wIDs[0]]['itemIndices']

		offsets = [studentScores[wID]['offsets'] for wID in wIDs]
		# try:
		calibratedFinalGrade, calibratedScore = alg.sumFunc(weights,scores,offsets,maxScoreVector)
		# Add entries to simulatedScores
		try:
			for i in range(len(itemIndices)):
				db.cursor.execute("INSERT INTO simulatedScores VALUES ((SELECT max(simulation) from simulations), ?, ?, ?,?)",[peerGroupID,itemIndices[i],calibratedScore[i],expertScore[i]])
		except:
			print('Could not calcluate grade:')
			print('Weights='+str(weights))
			print('Scores='+str(scores))
			print('Offsets='+str(offsets))
			print('MaxScores='+str(maxScoreVector))
			print('ExpertScore'+str(expertScore)) 
			print('itemIndices='+str(itemIndices))
			break
	db.conn.commit()

def generateSamplePeerGroups(db,Nmax,n,matchmaking='random'):
	# generates N list of groups of n students (peers) from among all the students who have graded ALL items of ALL expert-graded videos in ALL labs except labs 5 and 6

	totalN = getNgradedItems(db,1)+getNgradedItems(db,2)+getNgradedItems(db,3)+getNgradedItems(db,4)
	totalN = totalN*5 # FIXME: the total number of graded itemIndices needs to be multiplied by the number of expertvideos per lab
	db.cursor.execute('''SELECT r.wID, count(r.response)
			from responses r, experts e, rubrics b
			where
				r.URL = e.URL
				and r.itemIndex = e.itemIndex
				and r.labNumber = b.labNumber
				and r.itemIndex = b.itemIndex
				and b.graded 
				and r.labNumber < 5
				and r.response is not null 
				and r.response != ''
				GROUP BY r.wID order by r.wID''')
	data = [[str(entry[0]),int(entry[1])] for entry in db.cursor.fetchall()]
	validwIDs = [entry[0] for entry in data if entry[1]==totalN]
	# for wID, wIDgroup in groupby(data, key = lambda x: x[0]):
		# responses=[entry[1] for entry in wIDgroup]
	# if int(responses)==120:
	#     validwIDs.append(wID)
	# we now select N combinations at random from the list of all possible combinations of n validwIDs. If N is greater than the number of possible combinations, then we just use all possible combinations
	peerGroups = randomCombinations(validwIDs, Nmax, n)
	return peerGroups

def generateErrors(db,simulation):
	# generate all the square errors for each simulation (=a unique URL, N, n, and algorithm)
	db.cursor.execute('''INSERT INTO squareErrors (simulation, peerGroup, squareError) 
			SELECT simulation, peerGroup, SUM((simulatedScore - expertScore)*(simulatedScore-expertScore))
			FROM simulatedScores
			WHERE simulation = ?
			GROUP BY peerGroup''',[simulation])

def generateAbsErrors(db,simulation):
	# generate all the abs-value errors for each simulation (=a unique URL, N, n, and algorithm)
	db.cursor.execute('''INSERT INTO absErrors (simulation, peerGroup, absError) 
			SELECT simulation, peerGroup, SUM(ABS(simulatedScore - expertScore))
			FROM simulatedScores
			WHERE simulation = ?
			GROUP BY peerGroup''',[simulation])

def generateSignedErrors(db,simulation):
	# generate all the ±errors for each simulation (=a unique URL, N, n, and algorithm)
	db.cursor.execute('''INSERT INTO signedErrors (simulation, peerGroup, signedError) 
			SELECT simulation, peerGroup, SUM(simulatedScore - expertScore)
			FROM simulatedScores
			WHERE simulation = ?
			GROUP BY peerGroup''',[simulation])

def assignGrades(db,labNumber,algorithm,nCalibration = 3, term='F2014',test = False):
	# We want to calculate the final grade for a given URL
	# We must know which rubric items count toward the grade
	# We must know what each rating is worth for each item

	R = db.getNgradedItems(labNumber)
	# Gather the weights and scores of everyone who graded every URL submitted this lab, except the submitter
	maxScore, maxScoreVector = db.getMaxScore(labNumber)
	minScore, minScoreVector = db.getMinScore(labNumber)

	if algorithm.offsetStyle == None:
		db.cursor.execute('''SELECT r.URL, r.wID, r.itemIndex, w.weight, k.score, 0
			FROM studentEvaluations r, weights w, ratingKeys k, rubrics rub, submissions s 
			WHERE 
				--match URL
				r.URL is not null 
				AND r.URL = s.URL
				--match wID
				AND r.wID = w.wID
				AND r.wID != s.wID --don't include the submitter's rating
				--match labNumber
				AND r.labNumber = ? 
				AND w.labNumber = r.labNumber 
				AND k.labNumber = r.labNumber 
				AND rub.labNumber = r.labNumber 
				--match itemIndex
				AND w.itemIndex = r.itemIndex 
				AND rub.itemIndex = r.itemIndex 
				AND k.itemIndex = r.itemIndex 

				AND r.rating = k.rating 
				AND w.weightType = ? 
				AND w.nCalibration = ?
				AND rub.graded ORDER BY r.URL, r.wID, r.itemIndex''',[labNumber,algorithm.weightFunc.__name__,nCalibration])
	else:
		db.cursor.execute('''SELECT r.URL, r.wID, r.itemIndex, w.weight, k.score, off.weight 
			FROM studentEvaluations r, weights w, weights off, ratingKeys k, rubrics rub, submissions s 
			WHERE 
				--match URL
				r.URL is not null 
				AND r.URL = s.URL
				--match wID
				AND r.wID = w.wID
				AND r.wID = off.wID
				AND r.wID != s.wID --don't include the submitter's rating
				--match labNumber
				AND r.labNumber = ? 
				AND w.labNumber = r.labNumber 
				AND k.labNumber = r.labNumber 
				AND rub.labNumber = r.labNumber 
				AND off.labNumber = r.labNumber
				--match itemIndex
				AND w.itemIndex = r.itemIndex 
				AND rub.itemIndex = r.itemIndex 
				AND k.itemIndex = r.itemIndex 
				AND off.itemIndex = r.itemIndex

				AND r.rating = k.rating 
				AND w.weightType = ? 
				AND off.weightType = ?
				AND w.nCalibration = ?
				AND off.nCalibration = w.nCalibration
				AND rub.graded ORDER BY r.URL, r.wID, r.itemIndex''',[labNumber,algorithm.weightFunc.__name__ if algorithm.weightFunc != None else None,algorithm.offsetStyle,nCalibration])


	data = db.cursor.fetchall()
	data = list(data)
	print(len(data))
	URLlist = []
	# sort by URL
	for URL, URLweights in groupby(data, key = lambda x: str(x[0])):
		ratings = []
		# sort by responder wID
		for wID, wIDweights in groupby(list(URLweights), key=lambda x: str(x[1])):
			thisWeights = []
			thisScores = []
			thisOffsets = []
			for entry in list(wIDweights):
				# append every weight value
				thisWeights.append(float(entry[3]))
				thisScores.append(float(entry[4]))
				thisOffsets.append(float(entry[5]))
			if len(thisWeights) == len(thisScores) == len(thisOffsets) == R:
				ratings.append([wID, thisWeights, thisScores, thisOffsets])
		URLlist.append([URL,ratings])

	# Now we grade each URL
	for URLentry in URLlist:
		URL = URLentry[0]
		# print("Grading URL="+URL)
		ratings = URLentry[1]

		# Get the wID which submitted this video
		db.cursor.execute("SELECT wID FROM submissions WHERE labNumber = ? AND URL = ?",[labNumber,URL])
		data = db.cursor.fetchall()
		if len(data) != 1:
			print("URL "+URL+" belongs to zero or more than one people!")
		else:
			submitterwID = str(data[0][0])


		# Get the indices of the graded items for this rubric (TODO: we should only need to do this once per grade assignment)
		# TODO; do this when we're making the URLlist
		itemIndices = db.getScoresDict(labNumber).keys()

		weights = []
		scores = []
		offsets = []
		for entry in ratings:
			if len(entry[1]) == len(entry[2]) == len(entry[3]) == R:
				weights.append(entry[1])
				scores.append(entry[2])
				offsets.append(entry[3])

		# print(submitterwID)
		# print(URL)
		# for entry in URLlist:
		#     if entry[0]==URL:
		#         print(entry)
		# print(weights)
		# print(scores)
		# print(offsets)
		# 1/0
		# Send the weights, scores, and offsets to the sumFunc
		if len(weights) == len(scores) == len(offsets) >=1:
			finalGrade, finalGradeVector = algorithm.sumFunc(weights,scores,offsets=offsets,maxScores=maxScoreVector)

			# We need to correct for the case when any item in the finalGradeVector is not within the proper range for that item
			for i in range(len(finalGradeVector)):
				if minScoreVector[i] >= finalGradeVector[i] >= maxScoreVector[i]:
					pass
				else:
					finalGradeVector[i] = min(max(finalGradeVector[i],minScoreVector[i]),maxScoreVector[i])

			finalGrade = sum(finalGradeVector)
		# If all the graders have weight 0 for a particular item, we give the student the student's own grade instead. Don't make the SQLite query unless we have to.
		# TODO: we need to handle this case
		selfGrade = None
		for i in range(R):
			if False:
				# print('blarg!')
				if selfGrade == None:
					db.cursor.execute("SELECT score, ratings.itemIndex FROM ratings, ratingKeys, rubrics WHERE rubrics.labNumber = ratings.labNumber AND ratingKeys.labNumber = ratings.labNumber AND ratings.labNumber = ? AND wID = ? AND URL = ? AND ratings.itemIndex = ratingKeys.itemIndex AND rubrics.itemIndex = ratings.itemIndex AND rubrics.graded ORDER BY ratings.itemIndex",[labNumber,submitterwID,URL])
					data = [entry for entry in db.cursor.fetchall()]
					selfGradesDict = {}
					for entry in data:
						# itemIndex: selfScore
						selfGradesDict.update({ int(entry[1]): float(entry[0]) })
					# We have to make the selfGrade
					selfGrade = []
					# ...but we might not have to do the maxGrade
					maxGrade = None
					# someone might score some but not all of their own items, so we need to make sure that we handle each graded itemIndex explicitly rather than just the ordered selfGrade list
					for j in range(R):
						try:
							selfGrades.append(selfGradesDict[itemIndices[j]])
						except:
							selfGrade.append(None)
							# If the student didn't assign a grade to their own video, then we give them the max score for that item. Again, don't make the SQLite query unless we need to
							if maxGrade == None:
								# THIS ASSUMES rating=0 CORRESPONDS TO THE MAXIMUM SCORE
								db.cursor.execute("SELECT score FROM ratingKeys K, rubrics R WHERE K.labNumber= R.labNumber AND R.labNumber = ? AND rating = 0 AND R.itemIndex = K.itemIndex AND R.graded ORDER BY K.itemIndex",[labNumber])
								maxGrade = [float(entry[0]) for entry in db.cursor.fetchall()]
				elif selfGrade[i] != None:
					numerators[i] = selfGrade[i]
					rawNumerators[i] = selfGrade[i]
				else:
					numerators[i] = maxGrade[i]
					rawNumerators[i] = maxGrade[i]
				denominators[i] = 1
				rawDenominators[i] = 1

		if test:
				print(submitterwID+' new finalGrade: '+str(finalGrade))

		else:
		# Put the itemgrades in the itemgrades table, and the finalgrades in the finalgrades table
			try:

				for i in range(len(finalGradeVector)):
					db.cursor.execute("INSERT INTO itemGrades VALUES (NULL,?,?,?,?,?,?,?,?,?)",[datetime.now(),term,labNumber,submitterwID,URL,itemIndices[i],finalGradeVector[i],finalGradeVector[i]*100/maxScoreVector[i],algorithm.name])

				db.cursor.execute("INSERT INTO finalGrades VALUES (NULL,?,?,?,?,?,?,?,?)",[datetime.now(),term,labNumber, submitterwID, URL, finalGrade, finalGrade*100/maxScore, algorithm.name])
			except:
				print(submitterwID+' could not be graded. len(weights)='+str(len(weights))+' len(scores)='+str(len(scores))+' len(offsets)='+str(len(offsets)))
	if not test:
		db.conn.commit()

def printGradesReport(db, filename, labNumber, algorithm = 'offMean_1', weightType = 'weightDIBI_full'):
	maxScore = db.getMaxScore(labNumber)
	# R=Number of items in rubric
	R = db.getNgradedItems(labNumber)
	db.cursor.execute('''SELECT fg.wID, fg.grade, cg.grade
		FROM finalGrades fg, calibrationGrades cg
		WHERE 
			fg.wID = cg.wID 
			AND fg.labNumber = cg.labNumber 
			AND fg.labNumber = ? 
			AND fg.algorithm = ?
			AND cg.weightType = ?
			and cg.nCalibration = 3
			AND fg.grade IS NOT NULL 
			ORDER BY fg.wID
			''',[labNumber,algorithm,weightType])
	# db.cursor.execute('''SELECT cg.wID, 999, cg.grade
	#     FROM calibrationGrades cg
	#     WHERE
	#         cg.labNumber = ?
	#         AND cg.weightType = ?
	#         AND cg.grade IS NOT NULL
	#         AND cg.grade != 0
	#         AND cg.nCalibration = 3
	#         ORDER BY grade
	#     ''',[labNumber,weightType])
	with open(filename,'w') as output:
		# output.write('Student\tPresentation Grade\tCalibration Grade\n')
		data = db.cursor.fetchall()
		for entry in data:
			wID = str(entry[0])
			finalGrade = str(float(entry[1]))
			calibrationGrade = str(float(entry[2]))
			# output.write(wID+'\t'+calibrationGrade+'\n')
			output.write(wID+'\t'+finalGrade+'\t'+calibrationGrade+'\n')

# db = SqliteDB('S2014Campus.sqlite')
def printCalibrationGradesReport(db, filename, labNumber, weightType = 'weightDIBI_full'):
	maxScore = db.getMaxScore(labNumber)
	# R=Number of items in rubric
	R = db.getNgradedItems(labNumber)
	db.cursor.execute('''SELECT cg.wID, cg.calibrationGrade
		FROM calibrationGrades cg
		WHERE 
			cg.labNumber = ? 
			AND cg.weightType = ?
			and cg.nCalibration = 3
			ORDER BY cg.wID
			''',[labNumber,weightType])
	# db.cursor.execute('''SELECT cg.wID, 999, cg.grade
	#     FROM calibrationGrades cg
	#     WHERE
	#         cg.labNumber = ?
	#         AND cg.weightType = ?
	#         AND cg.grade IS NOT NULL
	#         AND cg.grade != 0
	#         AND cg.nCalibration = 3
	#         ORDER BY grade
	#     ''',[labNumber,weightType])
	with open(filename,'w') as output:
		# output.write('Student\tPresentation Grade\tCalibration Grade\n')
		data = db.cursor.fetchall()
		for entry in data:
			wID = str(entry[0])
			calibrationGrade = str(float(entry[1]))
			# output.write(wID+'\t'+calibrationGrade+'\n')
			output.write(wID+'\t'+calibrationGrade+'\n')
def printFinalGradesReport(db, filename, labNumber, algorithm = 'offMean_1'):
	maxScore = db.getMaxScore(labNumber)
	# R=Number of items in rubric
	R = db.getNgradedItems(labNumber)
	db.cursor.execute('''SELECT fg.wID, fg.grade
		FROM finalGrades fg
		WHERE 
			fg.labNumber = ? 
			AND fg.algorithm = ?
			ORDER BY fg.wID
			''',[labNumber,algorithm])
	# db.cursor.execute('''SELECT cg.wID, 999, cg.grade
	#     FROM calibrationGrades cg
	#     WHERE
	#         cg.labNumber = ?
	#         AND cg.weightType = ?
	#         AND cg.grade IS NOT NULL
	#         AND cg.grade != 0
	#         AND cg.nCalibration = 3
	#         ORDER BY grade
	#     ''',[labNumber,weightType])
	with open(filename,'w') as output:
		# output.write('Student\tPresentation Grade\tCalibration Grade\n')
		data = db.cursor.fetchall()
		for entry in data:
			wID = str(entry[0])
			finalGrade = str(float(entry[1]))
			# output.write(wID+'\t'+calibrationGrade+'\n')
			output.write(wID+'\t'+finalGrade+'\n')

