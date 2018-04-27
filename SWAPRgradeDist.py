from SWAPR3 import *
import sqlite3 as lite
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby

# algsToGraph = algNamesStrings
db = lite.connect('/Users/Scott/Research/SOUP 2017 SWAPR/SOUP2017.sqlite')
cur = db.cursor()

weightType = 'weightBIBI'
algorithm = 'offMean_1'
# labNumber = 4
plotYmax = 10

for labNumber in [4]:

	fig = plt.figure(figsize=(10,5))
	calibration = True
	if calibration: #Calibration Grades
		cur.execute('''SELECT calibrationGrade
		    FROM calibrationGrades
		    WHERE
		        labNumber = ?
		        and nCalibration = 3
		        AND weightType = ?
		        AND calibrationGrade > 0
		    ''',[labNumber,weightType])
		data = list(cur.fetchall())
		print(len(data))
		# Plot calibrationGrades
		ax = fig.add_subplot(1,3,1)
		grades = [float(entry[0]) for entry in data]
		nBins = 20
		ax.hist(grades,histtype='stepfilled',bins=[j*(100/nBins) for j in range(nBins+1)])
		ax.set_xlim([0,100])
		ax.set_ylim([0,plotYmax])
		ax.set_title('Lab '+str(labNumber)+' Calibration Grades\n('+weightType+')')
		ax.annotate('Mean='+str('%.2f' % np.mean(grades))+'\nStDev='+str('%.2f' % np.std(grades)), xy = (0.1, 0.9), verticalalignment = 'top', xycoords = 'axes fraction')
		ax.set_ylabel('Number of Students')
		ax.set_xlabel('Grade')


	cur.execute('''SELECT grade
		FROM finalGrades
		WHERE
			labNumber = ?
			AND algorithm = ?
		''',[labNumber,algorithm])
	data = list(cur.fetchall())
	# Plot calibrationGrades
	ax = fig.add_subplot(1,3,2)
	grades = [float(entry[0]) for entry in data]
	nBins = 20
	ax.hist(grades,histtype='stepfilled',bins=[j*(100/nBins) for j in range(nBins+1)])
	ax.set_xlim([0,100])
	ax.set_ylim([0,plotYmax])
	ax.set_title('Lab '+str(labNumber)+' Peer Grades\n('+algorithm+')')
	ax.annotate('Mean='+str('%.2f' % np.mean(grades))+'\nStDev='+str('%.2f' % np.std(grades)), xy = (0.1, 0.9), verticalalignment = 'top', xycoords = 'axes fraction')
	ax.set_xlabel('Grade')

	if calibration:
		cur.execute('''SELECT (0.33*(f.grade) + 0.33*c.calibrationGrade + 33)
			FROM finalGrades f, calibrationGrades c
			WHERE
				f.labNumber = ?
				AND f.labNumber = c.labNumber
				AND c.nCalibration = 3
				AND f.wID = c.wID
				AND f.algorithm = ?
				AND c.weightType = ?
			''',[labNumber,algorithm,weightType])
	else:
		cur.execute('''SELECT (0.66*(f.grade) + 33)
			FROM finalGrades f
			WHERE
				f.labNumber = ?
				AND f.algorithm = ?
			''',[labNumber,algorithm])	
	data = list(cur.fetchall())
	# Plot calibrationGrades
	ax = fig.add_subplot(1,3,3)
	grades = [float(entry[0]) for entry in data]
	nBins = 20
	ax.hist(grades,histtype='stepfilled',bins=[j*(100/nBins) for j in range(nBins+1)])
	ax.set_xlim([0,100])
	ax.set_ylim([0,plotYmax])
	if calibration:
		ax.set_title('Lab '+str(labNumber)+' Final Grades\n.33Calib + .33Part + .33Peer')
	else:
		ax.set_title('Lab '+str(labNumber)+' Final Grades\n.33Part + .66Peer')

	ax.annotate('Mean='+str('%.2f' % np.mean(grades))+'\nStDev='+str('%.2f' % np.std(grades)), xy = (0.1, 0.9), verticalalignment = 'top', xycoords = 'axes fraction')
	ax.set_xlabel('Grade')

	plt.tight_layout()
plt.savefig('Lab{}Grades.pdf'.format(labNumber))
plt.show()