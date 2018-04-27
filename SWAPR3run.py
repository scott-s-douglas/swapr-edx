# encoding: utf-8
# Used Summer 2017 for SOUP, PHYS2211
from SWAPR3 import *
from SWAPR3grades import *
import os

# ED: Download the student responses to the submission assignment from Webassign and put them
# makeDatabase('SOUP2017.sqlite')
db=SWAPRdb('/Users/Scott/Research/SOUP 2017 SWAPR/SOUP2017.sqlite')
# db.createTables()
# db.cursor.execute('''ATTACH DATABASE "/Users/Scott/SWAPR/S2014Campus_legacy.sqlite" AS old''')
# db.cursor.execute('''SELECT DISTINCT labNumber, questionIndex, wQuestion FROM old.questions''')
# data = [[entry[0], entry[1], entry[2]] for entry in db.cursor.fetchall()]
# for entry in data:
#     db.cursor.execute('''UPDATE assignments SET wQuestion = ? WHERE labNumber = ? AND questionIndex = ?''', [entry[2], entry[0], entry[1]])

# db.cursor.execute('''DELETE FROM studentEvaluations''')
# db.cursor.execute('''DELETE FROM submissions''')
for labNumber in [4]:
    # Define Algorithms
    calibAlgs = [
        calibAlg("BIBI_1",
            weightedSumMedianFallback,
            weightBIBI,
            longName = u"Binary Item-by-Item ±1"), # Used for grades in F2013
        calibAlg("median",
            sumMedian,
            longName = "Median"),
        calibAlg("mean",
            sumMean,
            longName = "Mean"),
        calibAlg("offMean_1",
            weightedSumOffset,
            weightBIBI,
            longName = u"Offset Mean ±1", offsetStyle = "weightOffset")
        ]

    # db.cursor.execute('''DROP TABLE IF EXISTS rubrics''')
    # db.cursor.execute('''DROP TABLE IF EXISTS studentEvaluations''')
    # db.cursor.execute('''DROP TABLE IF EXISTS weights''')
    # db.cursor.execute('''DROP TABLE IF EXISTS finalGrades''')
    # db.cursor.execute('''DROP TABLE IF EXISTS itemGrades''')
    # db.cursor.execute('''DROP TABLE IF EXISTS calibrationGrades''')
    # db.cursor.execute('''DELETE FROM submissions WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM expertEvaluations WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM assignments WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM weights WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM finalGrades WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM itemGrades WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM calibrationGrades WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM studentEvaluations WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM rubrics WHERE labNumber=?''',[labNumber])
    # db.cursor.execute('''DELETE FROM ratingKeys WHERE labNumber=?''',[labNumber])
    # db.createTables()
    # db.parseExpertEvaluations('F2014Lab'+str(labNumber)+'Experts.txt')
    # for file in listdir_nohidden('/Users/Scott/Research/SOUP 2017 SWAPR/Lab {}/Submissions/'.format(labNumber) ):
    #     print("Parsing "+str(file)+'...')
    #     db.parseSubmissions(file,labNumber,linkCol=4 if labNumber != 3 else 5,term='SOUP2017', verbose=True)
    # db.assignURLs(labNumber, term='SOUP2017')
    # db.exportWebassign('Lab{}Output.txt'.format(labNumber),labNumber)

    # db.addDefaultRubric(labNumber=labNumber,term="SOUP2016")
    # Gather evaluations and assign weights & grades
    for file in listdir_nohidden('/Users/Scott/Research/SOUP 2017 SWAPR/Lab {}/Responses/'.format(labNumber)):
        print("Parsing "+str(file)+'...')
        db.parseEvaluationsFile(file,labNumber,term='SOUP2017')
        
    # We need weightBIBI, weightDIBI_full, weightDIBI_full_curved, and weightOffset
    # assignWeights(db,labNumber=labNumber,f=weightDIBI_full)
    # assignWeights(db,labNumber=labNumber,f=weightDIBI_full_curved)
    # assignGrades(db,labNumber=labNumber,algorithm=calibAlgs[1])
    # assignGrades(db,labNumber=labNumber,algorithm=calibAlgs[2])

    assignWeights(db,labNumber=labNumber,f=weightBIBI)
    assignWeights(db,labNumber=labNumber,f=weightOffset)
    assignGrades(db,labNumber=labNumber,algorithm=calibAlgs[3])
    assignCalibrationGrades(db,labNumber=labNumber)
    printCalibrationGradesReport(db,'Lab'+str(labNumber)+'CalibrationGrades.txt',labNumber=labNumber,weightType='weightBIBI')
    printFinalGradesReport(db,'Lab'+str(labNumber)+'Grades.txt',labNumber=labNumber,algorithm='offMean_1')
    db.writeCommentsTabDelimited('Lab'+str(labNumber)+'Comments.txt',labNumber=labNumber)