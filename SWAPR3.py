# encoding: utf-8
import sqlite3
import os.path
import sys
import csv
import random
import glob
from itertools import groupby
from datetime import date, datetime
import re

def listdir_nohidden(path):
    # Return only the non-hidden files in a directory, to avoid that annoying .DS_Store file
    return glob.glob(os.path.join(path, '*'))

def getYoutubeID(string):
    # Get the YouTube ID from a URL or other string
    if string is not '' and string is not None:
        ID = re.search('[a-zA-Z0-9_-]{11}', string)
        if ID is not None:
            if len(ID.group(0)) == 11:
                return ID.group(0)
    return ''

def getCourseraID(string):
    # Get the Coursera ID from a URL or other string
    if string is not '' and string is not None:
        ID = re.search('(?<=lecture/)[0-9]{3}', string)
        if ID is not None:
            if len(ID.group(0)) == 3:
                return ID.group(0)
    return ''

def getedXID(string):
    # Get the edX ID from a URL or other string
    # Example: https://edge.edx.org/courses/course-v1:GTx+Phys2211+2016_Summer/courseware/8b8385c43251415aa38afd47300f0820/f252bea73742429783bd5b3f5610522c/
    # First /hash/ identifies the unit; second /hash/ identifies the subsection. We want to get both, along with the interior slash (not the trailing or leading slashes)
    if string is not '' and string is not None:
        ID = re.search('[a-zA-Z0-9]{32}/[a-zA-Z0-9]{32}', string)
        if ID is not None:
            if len(ID.group(0)) == 65:
                return ID.group(0)
    return ''    

def getYoutubeLink(string, verbose = False):
    # Turn any string which contains a valid YouTube ID into a youtu.be link
    if string is not '' and string is not None:
        if len(getYoutubeID(string)) == 11:
            return 'http://youtu.be/'+getYoutubeID(string)
        elif verbose:
            print("Rejecting YouTube URL '"+string+"'")    
    return ''

def perlPerson(string):   # Generate a Perl-safe STUDENT ID (i.e. no @ sign)
    return re.sub('@','_',string);

def getPerlLinksLine(wID, URLsToGrade):
    outstring = '\t\"' + perlPerson(wID) + '\"=> [ ';
    for video in URLsToGrade:
        # outstring += '\"' + getCourseraID(video) +'\", ';
        outstring += '\"' + getedXID(video) +'\", ';
    outstring += '],\n';
    return outstring
def makeDatabase(databaseName):
    '''Create a blank sqlite database'''
    if databaseName[-7:] != ".sqlite":
        databaseName = databaseName + ".sqlite"
    conn = sqlite3.connect(databaseName)
    conn.commit()
    conn.close()

class SWAPRdb:
    '''class for connecting, inserting, and retrieving information from a sqlite3 database'''   
    # connects to the database, alters its name if named incorrectly
    def __init__(self, databaseName):
        if databaseName[-7:] != ".sqlite":
            databaseName = databaseName + ".sqlite"
        if os.path.isfile(databaseName):
            self.databaseName = databaseName;
            self.conn = sqlite3.connect(self.databaseName)
            self.cursor = self.conn.cursor()
            self.conn.text_factory = str
        else:
            # sees if database name is unique, so it doesn't overwrite anything
            sys.exit("This database does not exist, use the makeDatabase(databaseName) to create it")

    def createTables(self):
        # create tables if they do not exist

        # student evaluations go here
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS studentEvaluations
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                wID text,
                labNumber int,
                URL text,
                videoLabel text,
                itemIndex int,
                graded int,
                rating int,
                comment text)
            ''')

        # student submissions go here
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS submissions
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                wID text,
                labNumber int,
                URL text,
                youtubeURL text,
                flag text, UNIQUE(wID,labNumber), UNIQUE(URL))
            ''')

        # expert evaluations go here
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS expertEvaluations
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                URL text,
                videoLabel text,
                vidOrder int,
                itemIndex int,
                rating int,
                comment text)
            ''')

        # student IDs and info go here
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS students
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                wID text,
                fullName text,
                email text,
                section text,
                flag text)
            ''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS rubrics
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                itemIndex int,
                topic text,
                body text,
                graded boolean)
            ''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS assignments
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                wID text,
                questionIndex int,
                wQuestion int,
                videoLabel text,
                URL text)
            ''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS ratingKeys
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                itemIndex int,
                rating int,
                ratingLabel int,
                score number)
        ''')
        self.conn.commit()

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS weights
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                nCalibration int,
                wID text,
                itemIndex int,
                weightType text,
                weight number)
        ''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS itemGrades
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                wID text,
                URL text,
                itemIndex int,
                itemScore number,
                itemGrade number,
                algorithm text)
        ''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS finalGrades
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                wID text,
                URL text,
                score number,
                grade number,
                algorithm text)
        ''')

        self.cursor.execute('''CREATE TABLE IF NOT EXISTS calibrationGrades
            (row INTEGER PRIMARY KEY NOT NULL,
                time timestamp,
                term text,
                labNumber int,
                wID text,
                nCalibration int,
                calibrationScore number,
                calibrationGrade number,
                weightType text)
        ''')

    def writeCommentsTabDelimited(self,filename,labNumber,algorithm='offMean_1',weightType = 'weightDIBI_full_curved',writeEmails = False,reportWeights = True, reportGrades = True):

        itemIndices=[1,2,3,4,5]

        with open(filename,'w') as output:
            labelString = "Username"
            if writeEmails:
                labelString += "\tEmail"
            labelString += "\tURL"
            # for i in range(5):
            # TODO: don't hard-code the number of studentEvaluations
            if reportGrades:
                for i in range(len(itemIndices)):
                    labelString+="\tItem "+str(i+1)+"\tItem "+str(i+1)+" Grade\tItem "+str(i+1)+" Comments"
                labelString += "\n"
                output.write(labelString)
            else:
                for i in range(len(itemIndices)):
                    labelString+="\tItem "+str(i+1)+"\tItem "+str(i+1)+" Comments"
                labelString += "\n"
                output.write(labelString)
            # Get the list of item prompts
            # self.cursor.execute("SELECT itemPrompt FROM rubrics WHERE labNumber = ? AND itemType = 'freeResponse' AND itemIndex != 14 ORDER BY itemIndex",[labNumber])
            # Get the list of item prompts (5-item extra credit rubric)
            self.cursor.execute("SELECT topic FROM rubrics WHERE labNumber = ? AND itemIndex != 6 ORDER BY itemIndex",[labNumber])
            prompts = [str(prompt[0]) for prompt in self.cursor.fetchall()]

            self.cursor.execute('''SELECT submissions.wID, submissions.URL, studentEvaluations.itemIndex, comment
                FROM studentEvaluations, submissions
                WHERE 
                    submissions.URL = studentEvaluations.URL 
                    and submissions.labNumber = ? 
                    AND studentEvaluations.labNumber = submissions.labNumber 
                    AND studentEvaluations.itemIndex != 6 
                    ORDER BY submissions.wID, studentEvaluations.itemIndex''',[labNumber])
            data = [[str(entry[0]),str(entry[1]),int(entry[2]),str(entry[3])] for entry in self.cursor.fetchall() if entry[3] != None]
            for wID, studentEvaluations in groupby(data,key = lambda x: x[0]):
                studentEvaluations = list(studentEvaluations)
                URL = studentEvaluations[0][1]
                # Get the student's peers' comments
                peerComments = {}
                for itemIndex, itemstudentEvaluations in groupby(studentEvaluations,key = lambda x: x[2]):
                    thisstudentEvaluations = ''
                    for entry in itemstudentEvaluations:
                        thisstudentEvaluations+=entry[3]+'; '
                    peerComments.update({itemIndex: thisstudentEvaluations})
                if reportWeights:
                    # self.cursor.execute("SELECT weight1, weight2, weight3, weight4, weight5, weight6 FROM weightsBIBI WHERE wID = ? and labNumber = ?",[wID, labNumber])
                    # weights = [[float(d[i]) for i in range(6)] for d in self.cursor.fetchall()]
                    self.cursor.execute('''SELECT weight
                        FROM weights
                        WHERE 
                            wID = ?
                            AND labNumber = ?
                            AND weightType = ?
                            AND nCalibration = 3
                        ORDER BY wID,itemIndex
                            ''',[wID,labNumber,weightType])
                    weights = [float(entry[0]) for entry in self.cursor.fetchall()]

                # Get the student's grade vector
                if reportGrades:
                    self.cursor.execute("SELECT itemScore FROM itemGrades WHERE wID = ? and itemGrades.labNumber = ? AND algorithm = ? ORDER BY itemIndex",[wID, labNumber,algorithm])
                    gradeVector = [item for item in self.cursor.fetchall()]
                    if gradeVector == []:
                        gradeVector = [0,0,0,0,0]
                    else:
                        gradeVector = [float(entry[0]) for entry in gradeVector]

                # Get email, if appropriate
                if writeEmails:
                    hasEmail = checkEmail(self, wID)
                    if hasEmail:
                        self.cursor.execute("SELECT email FROM students WHERE wID = ?",[wID])
                        email = str(self.cursor.fetchone()[0])
                    # print(email)
                dataString = ''
                if writeEmails:
                    if hasEmail:
                        try:
                            dataString += email
                            dataString += '\t'
                        except:
                            dataString += '\t'
                dataString += wID.split('@')[0]
                dataString += '\t'
                if URL is not None:
                    dataString += URL
                # for i in range(6):
                # TODO: don't hard code this
                for index in itemIndices:
                    i = itemIndices.index(index)
                    # if len(peerComments) > 0:
                        # for comment in peerComments:
                        # print(peerComments)
                        #   iComments += comment[i]+'; '
                    dataString += '\t'+prompts[i]+'\t'
                    if reportGrades:
                        dataString+=str('%.2f'%gradeVector[i])+'/20\t'
                    try:
                        dataString+=str(peerComments[index])
                    except: 
                        dataString+='(No peer comments)'
                    # else:
                    #   dataString += '\t'+prompts[i]+'\t'+str('%.2f'%gradeVector[i])+'%\t'+''
                    # try:
                    #   dataString += '\t'+str('%.2f' % (weights[i]*100))+'%'
                    # except:
                    #   dataString += '\t'+''
                dataString+='\n'

                output.write(dataString)

    def getScoresDict(self,labNumber):
        # Construct a dictionary of dictionaries where each possible response is paired with its score for GRADED items only
        self.cursor.execute('''SELECT k.itemIndex, k.response, k.score 
            FROM responseKeys k, rubrics r 
            WHERE 
                --match labNumber
                r.labNumber = ?
                AND r.labNumber = k.labNumber 
                --match itemIndex
                AND r.itemIndex = k.itemIndex
                AND k.score IS NOT NULL 
                AND r.graded
            ORDER BY k.itemIndex, k.response, k.score''',[labNumber])
        data = [[int(entry[0]),int(entry[1]),float(entry[2])] for entry in self.cursor.fetchall()]
        scoresDict = {}
        for itemIndex, itemIndexGroup in groupby(data, lambda entry: entry[0]):
            thisScores = {}
            for pair in itemIndexGroup:
                thisScores.update({pair[1]:pair[2]})
            scoresDict.update({itemIndex:thisScores})
        return scoresDict

    def getMaxScore(self,labNumber):
        self.cursor.execute('''SELECT max(score) 
            FROM ratingKeys, rubrics 
            WHERE 
                ratingKeys.labNumber = ? 
                AND ratingKeys.itemIndex = rubrics.itemIndex 
                AND ratingKeys.labNumber = rubrics.labNumber 
                AND graded
                GROUP BY rubrics.itemIndex
                ''',[labNumber])
        maxScoreVector = [float(entry[0]) for entry in self.cursor.fetchall()]
        maxScore = sum(maxScoreVector)

        return maxScore, maxScoreVector

    def getMinScore(self,labNumber):
        self.cursor.execute('''SELECT min(score) 
            FROM ratingKeys, rubrics 
            WHERE 
                ratingKeys.labNumber = ? 
                AND ratingKeys.itemIndex = rubrics.itemIndex 
                AND ratingKeys.labNumber = rubrics.labNumber 
                AND graded
                GROUP BY rubrics.itemIndex
                ''',[labNumber])
        minScoreVector = [float(entry[0]) for entry in self.cursor.fetchall()]
        minScore = sum(minScoreVector)

        return minScore, minScoreVector

    def getNgradedItems(self,labNumber):
        "Return the number of graded items in a particular lab's rubric. This function makes a SQLite call, so don't run it between a select and a fetch on that same database."
        self.cursor.execute('''SELECT count(*)
            FROM rubrics
            WHERE
                labNumber = ?
                AND graded
            ''',[labNumber])
        Ngraded = int(self.cursor.fetchone()[0])
        return Ngraded

    def addStudentSubmission(self,wID,youtubeURL,labNumber,URL=None,term='SOUP2016'):
        # Try to add a single student's URL submission. Flags the appropriate entry when the student already submitted a URL that lab, or if the URL already exists
        successState = False

        try:
            self.cursor.execute('''INSERT INTO submissions(row,time,term,wID,labNumber,URL,youtubeURL) VALUES (NULL,?,?,?,?,?,?)''',
            [datetime.now(),term,wID,labNumber,URL,youtubeURL])
            successState = True
        except sqlite3.IntegrityError as error:

            # One of the uniqueness contraints has been violated
            if error.message == 'UNIQUE constraint failed: submissions.wID, submissions.labNumber':
                # student already submitted this lab
                self.cursor.execute('''UPDATE submissions SET flag = '>1 URL this lab' WHERE wID = ? and labNumber = ?''',[wID,labNumber]) # flag the student's existing submission
                self.cursor.execute('''SELECT row FROM submissions WHERE wID = ? and labNumber = ?''',[wID,labNumber])
                row = self.cursor.fetchone()[0]
                print('WARNING: '+wID+' attempting to submit another URL for Lab '+str(labNumber)+'; flagging row '+str(row))

            elif error.message == 'UNIQUE constraint failed: submissions.URL':
                # URL has already been submitted
                self.cursor.execute('''UPDATE submissions SET flag = 'duplicate URL' WHERE URL=?''',[URL]) # flag the existing URL
                self.cursor.execute('''SELECT row FROM submissions WHERE URL = ?''',[URL])
                row = self.cursor.fetchone()[0]
                print('WARNING: '+wID+' attempting to submit existing URL, '+URL+'; flagging row '+str(row))
        return successState

    def addStudent(self,wID,term,section='N07'):
        # Add every student to the student table, if that student does not already exist. Each student must be unique. This table is just intended to serve as a store of students' emails, sections, etc., and not directly involved in error checking.
        # 17Aug2014, expects students to be gatech students in PHYS 2211 section N07
        try:
            self.cursor.execute('''INSERT INTO students(row,time,term,wID,email,section) VALUES (NULL,?,?,?,?,?)
                ''',[datetime.now(),term,wID,wID+'.edu',])
        except:
            print('Unable to add '+wID+' to '+term+' students list.')

    def validateStudents(self):
        # If a student switches sections mid-term, they can appear twice in the students table. Such instances will be flagged for manual resolution. Students may also take the class twice, in which case duplicate wIDs will appear in different terms; this is fine.
        self.cursor.execute('''UPDATE students SET flag = 'in >1 section this term' WHERE wID in 
            (SELECT DISTINCT wID FROM students GROUP BY wID, term HAVING count(wID) >1)''')
        

    def parseSubmissions(self,inputFile,labNumber,verbose=False,wIDcol=1,linkCol=4, term='SOUP2016'):
        # Reads a Webassign-formatted tab-delimited .csv file containing YouTube links submitted by students.
        #
        # When you download the .csv file, you must inclide the student responses, not just their scores. Scores are optional; the parser knows to ignore them.

        data = []
        with open(inputFile, 'rU') as csvfile:
            inputFile = csv.reader(csvfile, delimiter='\t', quotechar='"')
            # TODO: change this to operate directly on the file itself, not just blindly copy the whole file into memory
            for row in inputFile:
                data.append(row)

        line = 0
        foundStart = False
        while not foundStart:    #Seek out the first line of student data
            if len(data[line]) >= 1:
                if data[line][0] == 'Fullname': # The data starts right after the header line beginning with 'Fullname'
                    foundStart = True
            line = line+1


        # Go through the data line-by-line, and add students + URLs to the database
        while line in range(len(data)):
            if len(data[line]) > 1: # Make sure we don't have a blank line
                if data[line][0] != '': # Make sure we don't have one of the score lines
                    wID=data[line][wIDcol]
                    youtubeURL=getYoutubeLink(data[line][linkCol])
                    if youtubeURL not in ['',None]:
                        # if skipLinkless, then we won't add people who haven't submitted links
                        if verbose:
                            print('Adding submission: wID='+wID+', URL='+youtubeURL+', labNumber='+str(labNumber))
                        if youtubeURL == '':
                            youtubeURL = None
                        self.addStudentSubmission(wID,youtubeURL,labNumber,term=term)
            line += 1
        self.conn.commit()

    def parseExpertEvaluations(self,filename,term='F2014',wIDcol=1):
        # We write the experts file ourselves, so we use a less irritating format than the WebAssign .csv's we have to use for the student data
        data = []
        reportEntries = []
        with open(filename, 'rU') as csvfile:
            inputFile = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in inputFile:
                data.append(row)

        line = 1    # Data starts on the second line

        while line in range(len(data)):
            if len(data[line]) > 1: # Make sure we don't have a blank line
                if data[line][0][0] is not '#': # Make sure we don't have a commented line
                    labNumber = data[line][0]
                    label = data[line][1]
                    URL = data[line][2]
                    order = data[line][3]
                    itemIndex = data[line][4]
                    rating = data[line][5]
                    comment = data[line][6]
                    self.cursor.execute("INSERT INTO expertEvaluations(row,time,term,labNumber,videoLabel,URL,vidOrder,itemIndex,rating,comment) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [datetime.now(), term, labNumber, label, URL, order, itemIndex, rating, comment])
            line += 1
        self.conn.commit()
        return reportEntries
        
    def getURLsToGrade(self, wID, labNumber):
        self.cursor.execute("Select URL FROM assignments WHERE wID=? and labNumber=? ORDER BY questionIndex", [wID, labNumber])
        dbExtract = [entry for entry in self.cursor.fetchall()]
        if dbExtract == []:
            return False
        else:
            return [str(i[0]) for i in dbExtract]

    def exportWebassign(self,filename,labNumber):

        self.cursor.execute("SELECT DISTINCT wID FROM assignments WHERE labNumber = ?",[labNumber])
        wIDs = [str(item[0]) for item in self.cursor.fetchall()]

        self.cursor.execute("SELECT count(DISTINCT URL) FROM expertEvaluations WHERE labNumber = ? AND videoLabel NOT LIKE 'Hidden%'",[labNumber])
        evalStartIndex = int(self.cursor.fetchone()[0]) # The webassign evaluation assignment will contain URLs from getURLsToGrade starting with the evalStartIndex-th entry; this excludes the practice and unhidden calibration videos
        # print('evalStartIndex='+str(evalStartIndex))

        with open(filename,'w') as output:
            output.write('<eqn>\n'
                '#!/usr/bin/env perl\n'
                '%linkdb = (\n')

            for wID in wIDs:
                # if self.getURL(wID,labNumber) not in ['',None]:
                output.write(getPerlLinksLine(wID,self.getURLsToGrade(wID,labNumber)[evalStartIndex:]))
            output.write(');\n\n')

            output.write('sub get_link {\n'
                'my ($stu, $linknum) = @_;\n'
                'if ($linkdb{$stu}[$linknum]) {'
                'return $linkdb{$stu}[$linknum];\n'
                '} else {\n'
                'return $linkdb{"default"}[$linknum]}\n'
                "}\n''\n"
                '</eqn>\n'
                '<eqn>\n'
                '$this_student = $STUDENT;'  # We need to strip out the @
                '$this_student =~ s/@/_/g;'  # because that breaks hash lookup
                '\n'
                "'';"  # Make sure the output of this expression is blank
                '</eqn>\n\n')

            # output.write('<b>Please watch and respond to the following video:</b> <a href=http://youtu.be/<EQN>get_link($this_student,$QUESTION_NUM-1);</EQN> target="_blank"><eqn get_link($this_student,$QUESTION_NUM-1);></a> (This might be your own video; if so, please grade it honestly!)\n<br><br>\n\n')


    def assignURLs(self,labNumber,Npeer=3,term='F2014'):
        # construct a list of all wID, URL pairs for every submission in this lab, ordered by URL (this ends up being pseudorandom)
        # students who DID NOT submit their own URL this lab will all be assigned the same set of URLs to evaluate. This set will also be assigned to exactly one student who DID submit their own URL. Thus, each URL will be assigned to AT LEAST Npeer students, Npeer of whom DID submit their own URL for that lab.
        self.cursor.execute("SELECT DISTINCT wID, URL FROM submissions WHERE labNumber = ? AND URL IS NOT NULL ORDER BY URL",[labNumber])
        submissionList = [[str(entry[0]),str(entry[1])] for entry in self.cursor.fetchall()]

        # practice expert URLs
        self.cursor.execute("SELECT DISTINCT URL FROM expertEvaluations WHERE labNumber = ? AND videoLabel LIKE 'Practice%'",[labNumber])
        practiceList = [str(entry[0]) for entry in self.cursor.fetchall()]

        # expert URLs to be presented as calibration URLs
        self.cursor.execute("SELECT DISTINCT URL FROM expertEvaluations WHERE labNumber = ? AND videoLabel LIKE 'Calibration%'",[labNumber])
        shownList = [str(entry[0]) for entry in self.cursor.fetchall()]

        # expert URLs to be hidden among the peer URLs
        self.cursor.execute("SELECT DISTINCT URL FROM expertEvaluations WHERE labNumber = ? AND videoLabel = 'Hidden Calibration'",[labNumber])
        hiddenList = [str(entry[0]) for entry in self.cursor.fetchall()]

        # URL assignment for every student who submitted a video
        for i in range(len(submissionList)):
            URLsToGrade = []
            wID = submissionList[i][0]

            # add all the hidden expert URLs
            for URL in hiddenList:
                URLsToGrade.append(URL)

            j = 0
            while len(URLsToGrade) < Npeer + len(hiddenList) + 1: # the +1 is for the student's own video
                # add the student's own URL and the next Npeer URLs, wrapping around to the beginning of the list, skipping over URLs when they're already present in URLsToGrade. Will fail if len(submissionList) <= Npeer
                thisURL = submissionList[(i+j)%len(submissionList)][1]
                if thisURL not in URLsToGrade:
                    URLsToGrade.append(submissionList[(i+j)%len(submissionList)][1])
                j += 1
                # safety valve
                if j > len(submissionList):
                    break
            random.seed(wID)
            random.shuffle(URLsToGrade)

            # add the practice and then the shown URLs, in order, at the beginning
            URLsToGrade = practiceList + shownList + URLsToGrade

            for j in range(len(URLsToGrade)):
                self.cursor.execute("INSERT INTO assignments(row,time,term,labNumber,wID,questionIndex,wQuestion,videoLabel,URL) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",[datetime.now(),term,labNumber,wID,j+1,None,None,URLsToGrade[j]])
        # repeat the last set of URLsToGrade as the default set for students who didn't submit a video.
        for j in range(len(URLsToGrade)):
            self.cursor.execute("INSERT INTO assignments(row,time,term,labNumber,wID,questionIndex,wQuestion,videoLabel,URL) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",[datetime.now(),term,labNumber,'default',j+1,None,None,URLsToGrade[j]])
        self.conn.commit()

    def parseEvaluationsFile(self,filename,labNumber,term='F2014',wIDcol=1):
        # parse student responses from the associated Webassign .csv file. As of October 2013, there are 3 such files associated with each lab; a practice file and a calibration file which each contains only fixed expert-graded URLs, and an evaluation file containing shuffled student URLs, the student's own URL, and one expert URL.

        # Regardless of which file we feed it, this function matches the student's responses up with the appropriate URL by mapping the given Webassign Question ID to that response's question index, and then mapping the question index to that student's URLsToGrade
        data = []
        R = self.getNgradedItems(labNumber)
        with open(filename, 'rU') as csvfile:
            inputFile = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in inputFile:
                data.append(row)


            line = 0

            questions = []
            foundStart = False

            questionCols = {}   # a dictionary of {column: wQuestion} which shows which column corresponds to the start of the responses to which wQuestion
            isMultistep = False
            # New feature as of Lab 3 S2014; students can flag assignments as unwatchable, with the following error cordes: 0) No error; 1) Video is private, missing, or unavailable; 2) Profound technical problems like missing audio; 3) This video isn't the one the presenter intended to upload
            # NEW new feature as of SOUP 2016; we've dispensed with the multistep problems, they were too complicated for their own good.
            isFlaggable = False
            isScaffold = False
            # Multistep questions are NEVER flaggable
            while not foundStart:    #Seek out the first line of student data
                if len(data[line]) >= 1:
                    # If this file is from a practice or calibration assignment, then the responses are 2-step
                    if data[line][0] == 'Assignment Name':
                        for word in data[line][1].split(' '):
                            if word in ['Practice']:
                                # isMultistep = True
                                isScaffold = True
                                print(filename+" is a scaffold assignment.")
                            elif word in ['Evaluation']:
                                isFlaggable = True
                                print(filename+" is a flaggable assignment.")
                    if data[line][0] == "Questions":
                        for entry in data[line]:
                            if re.match('[0-9]',entry):    # Check to see if we've got a question number
                                questionCols.update({data[line].index(entry): int(entry)})
                                questions.append(int(entry))
                    if data[line][0] == 'Fullname':
                        foundStart = True
                line = line+1
                # print(questionCols)

            # Down in the student grade data, now
            while line in range(len(data)):
                if len(data[line]) > 1: # Make sure we don't have a blank line
                    if data[line][0] != '': # Make sure we don't have one of the score lines
                        # Now we have a student in the grade file; find the right student in self
                        wID = data[line][wIDcol]
                        fullName = data[line][0]
                        # print("Parsing responses for "+wID+'...')

                        # Now we see if this is line is from a multipart practice, disclosed) question or not (peer, self, hidden)



                        for qStart in questionCols.keys():  # go over every question
                            # qStart = colNum  # this question starts at this column
                            j = 0
                            wQuestion = questionCols[qStart]
                            # self.cursor.execute("SELECT assignments.URL FROM assignments, questions WHERE assignments.questionIndex = questions.questionIndex AND questions.wQuestion = ? AND assignments.labNumber = questions.labNumber AND assignments.labNumber = ? AND assignments.wID = ?",[wQuestion,labNumber, wID]) # Pre-F2014
                            self.cursor.execute("SELECT assignments.URL, assignments.videoLabel FROM assignments WHERE assignments.wQuestion = ? AND assignments.labNumber = ? AND assignments.wID = ?",[wQuestion,labNumber, wID])


                            try:
                                URL, videoLabel = self.cursor.fetchone()
                            # If the student didn't submit a URL themselves, then they got the 'default' URL assignments
                            except:
                                # self.cursor.execute("SELECT assignments.URL FROM assignments, questions WHERE assignments.questionIndex = questions.questionIndex AND questions.wQuestion = ? AND assignments.labNumber = questions.labNumber AND assignments.labNumber = ? AND assignments.wID = ?",[wQuestion,labNumber, 'default']) # Pre-F2014
                                self.cursor.execute("SELECT assignments.URL FROM assignments WHERE assignments.wQuestion = ? AND assignments.labNumber = ? AND assignments.wID = ?",[wQuestion,labNumber, 'default'])
                                URL = self.cursor.fetchone()
                                videoLabel = 'Peer Evaluation'
                                try:
                                    URL = str(URL[0])
                                except:
                                    print('Response has invalid URL: URL='+str(URL)+', wID='+wID+', wQuestion='+str(wQuestion))
                            # Break the line up into chunks of KNOWN LENGTH starting at colNum
                            # TODO: This is messy; info about the structure of the WebAssign question should be encoded into the rubric

                            # As of 5Mar2013, the practice + disclosed calibration videos have 2 steps; the first step has R graded numerical responses + 1 ungraded numerical response, and the second step has R sequential graded numerical/ungraded comment pairs and then 1 ungraded numerical response.

                            # As of 27Mar2013, the evaluation questions also include a numerical error response at the very end.

                            # First, get the responses for the first step
                            if isMultistep:
                                firstStepResponses = {}
                                lenFirstStep = R+1
                                for i in range(lenFirstStep):
                                    itemIndex = 2*i + 1 # the first step has no comments
                                    response = data[line][i+qStart]
                                    # blank responses should be Nones, not empty strings
                                    if response == '':
                                        response = None
                                    firstStepResponses.update({itemIndex: response})

                                # Now for the second (confirmation) step
                                secondStepResponses = {}
                                lenSecondStep = 2*R + 1
                                for i in range(lenSecondStep):
                                    itemIndex = i + 1 # second step has comments
                                    try:
                                        response = data[line][i + lenFirstStep + qStart]
                                    except:
                                        print('lenSecondStep = '+str(lenSecondStep))
                                        print('qStart = '+str(qStart))
                                        print('column = '+str(i + lenFirstStep + qStart))

                                    # blank responses should be Nones, not empty strings
                                    if response == '':
                                        response = None
                                    secondStepResponses.update({itemIndex: response})

                                # We take the second step responses. If any of those numerical responses are blank, then we use the corresponding response from the first step
                                finalResponses = {}
                                for itemIndex in secondStepResponses.keys():
                                    if secondStepResponses[itemIndex] == None:
                                        try:
                                            finalResponses.update({itemIndex: firstStepResponses[itemIndex] })
                                        except KeyError:
                                            finalResponses.update({itemIndex: None})
                                    else:
                                        finalResponses.update({itemIndex: secondStepResponses[itemIndex]})
                            else:
                                finalResponses = {}
                                if isFlaggable:
                                # If the question is a flaggable question, then the final (non-graded) numerical response also has a comment.
                                    for i in range(2*(R + 1)):
                                        # if line == 20 or line == 21:
                                            # print('Line '+str(line+1)+' Length '+str(len(data[line])))
                                            # print('i '+str(i))
                                            # print(data[line])
                                        itemIndex = i + 1
                                        response = data[line][i + qStart]
                                        finalResponses.update({itemIndex: response})
                                    try:
                                        flag = int(data[line][qStart + i + 1])
                                    except:
                                        flag = None
                                        # Flags not properly implemented
                                        # self.cursor.execute('''INSERT INTO flags VALUES (NULL, ?, ?, ?, ?)''',[labNumber, wID, URL, flag])
                                if isScaffold:
                                # If the question is a scaffold question, the final (non-graded) numerical response does not have a comment.
                                    for i in range(2*R + 1):
                                        # if line == 20 or line == 21:
                                            # print('Line '+str(line+1)+' Length '+str(len(data[line])))
                                            # print('i '+str(i))
                                            # print(data[line])
                                        itemIndex = i + 1
                                        response = data[line][i + qStart]
                                        finalResponses.update({itemIndex: response})



                            # print('Response: '+wID+' '+URL+' '+str(finalResponses))
                            # print('First Response: '+str(firstStepResponses))
                            # print('Second Response: '+str(secondStepResponses))
                            # 1/0

                            # Now we need to refactor finalResponses to separate the rating and the comment
                            refactoredResponses = {}
                            for itemIndex in finalResponses.keys():
                                if itemIndex%2 == 1:
                                    realItemIndex=(itemIndex+1)/2
                                    try:
                                        comment = finalResponses[itemIndex+1]
                                    except:
                                        comment = None
                                    refactoredResponses.update({realItemIndex:[finalResponses[itemIndex],comment]})

                            for itemIndex in refactoredResponses.keys():
                                rating = refactoredResponses[itemIndex][0]
                                comment = refactoredResponses[itemIndex][1]
                                # try:
                                self.cursor.execute("INSERT INTO studentEvaluations VALUES(NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [datetime.now(), term, wID, labNumber, URL, videoLabel, itemIndex, None, rating, comment])
                                # except:
                                #     print("Non-Unicode string in "+filename+" line "+str(line)+' '+comment)

                line += 1
            self.conn.commit()

    def addRubricItem(self, term, labNumber, itemIndex, topic, body=None, graded=True, itemValues = []):
        self.cursor.execute("INSERT INTO rubrics VALUES (NULL, ?, ?, ?, ?, ?, ?, ?)", [datetime.now(), term, labNumber, itemIndex, topic, body, graded])
        if itemValues != []:
            for i in range(len(itemValues)):
                self.cursor.execute("INSERT INTO ratingKeys VALUES (NULL, ?,?,?,?,?,?,?)",[datetime.now(), term, labNumber, itemIndex, i, itemValues[i][1], float(itemValues[i][0])])
        self.conn.commit()


    def addDefaultRubric(self, labNumber, term='F2014'):
        self.addRubricItem(labNumber = labNumber, itemIndex = 1, term=term, itemValues= [[11,'Poor'],[15,'Fair'],[17,'Good'],[19,'Very Good'],[20,'Excellent']], topic= 'Organization Structure')
        self.addRubricItem(labNumber=labNumber, itemIndex = 2, term = term, itemValues=[[11,'Poor'],[15,'Fair'],[17,'Good'],[19,'Very Good'],[20,'Excellent']], topic='Content Models')
        self.addRubricItem(labNumber=labNumber, itemIndex=3, term=term, itemValues=[[11,'Poor'],[15,'Fair'],[17,'Good'],[19,'Very Good'],[20,'Excellent']], topic='Content Prediction Discussion')
        self.addRubricItem(labNumber=labNumber, itemIndex=4, term=term, itemValues=[[11,'Poor'],[15,'Fair'],[17,'Good'],[19,'Very Good'],[20,'Excellent']], topic='Content Overall')
        self.addRubricItem(labNumber=labNumber, itemIndex=5, term=term, itemValues=[[14,'Poor'],[16,'Fair'],[18,'Good'],[19,'Very Good'],[20,'Excellent']], topic='Production Delivery')
        self.addRubricItem(labNumber=labNumber, itemIndex=6, term=term, itemValues=[[2,'Much better than mine'],[1,'Better than mine'],[0,'As good as mine'],[-1,'Worse than mine'],[-2,'Much worse than mine']], topic='How does this video compare to your video?',graded=False)

    def getScoresDict(self,labNumber):
        # Construct a dictionary of dictionaries where each possible response is paired with its score for GRADED items only
        self.cursor.execute('''SELECT k.itemIndex, k.rating, k.score 
            FROM ratingKeys k, rubrics r 
            WHERE 
                --match labNumber
                r.labNumber = ?
                AND r.labNumber = k.labNumber 
                --match itemIndex
                AND r.itemIndex = k.itemIndex
                AND k.score IS NOT NULL 
                AND r.graded
            ORDER BY k.itemIndex, k.rating, k.score''',[labNumber])
        data = [[int(entry[0]),int(entry[1]),float(entry[2])] for entry in self.cursor.fetchall()]
        scoresDict = {}
        for itemIndex, itemIndexGroup in groupby(data, lambda entry: entry[0]):
            thisScores = {}
            for pair in itemIndexGroup:
                thisScores.update({pair[1]:pair[2]})
            scoresDict.update({itemIndex:thisScores})
        # print(scoresDict)
        return scoresDict

    # Run the submission/assignment routine
    def submitAssign(self,inputFile,outputFile,labNumber):
        labNumber = self.parseSubmissions(inputFile)
        self.assignURLs(labNumber)
        self.writeAssignments(outputFile,labNumber)

    # Run the evaluation collection/grade distribution routine
    def collectGrade(self,inputFile,outputFile):
        labNumber = self.parseEvaluations(inputFile)
        self.assignWeights(labNumber)
        self.assignGrades(labNumber)
        self.writeGrades(outputFile,labNumber)
        self.writeFinalEvaluations(outputFile+'FinalEvaluations',labNumber)


        # 
# makeDatabase('test.sqlite')