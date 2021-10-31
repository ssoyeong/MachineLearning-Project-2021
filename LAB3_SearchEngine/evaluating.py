import re
import os

def evaluation (keyFileName, responseFileName, k, e, total_queries=225,total_documents=1400):
	keyFile = open(keyFileName, 'r')
	key = keyFile.readlines()
	key_dict = {}
	responseFile = open(responseFileName, 'r')
	response = responseFile.readlines()
	response_dict = {}
	all_precisions = []
	all_recalls = []
	missing_responses = []
	for line in key:
		line = line.rstrip(os.linesep)
		line = line.rstrip(' ')
		query,abstract,score = re.split(' +',line)
		query = int(query)
		abstract = int(abstract)
		score = int(score)
		if abstract <= total_documents:
			if query in key_dict:
				if not abstract in key_dict[query]:
					key_dict[query].append(abstract)
			else:
				key_dict[query] = [abstract]

	for line in response:
		line = line.rstrip(os.linesep)
		line = line.rstrip(' ')
		query,abstract,score = re.split(' +',line)
		if not (query.isdigit() and abstract.isdigit() and re.search('^[0-9\.-]+$',score)):
			print('Warning: Each line should consist of 3 numbers with a space in between')
			print('This line does not seem to comply:',line)
			exit()
		query = int(query)
		abstract = int(abstract)
		score = float(score)
		if (e < score): # parameter1
			if query in response_dict:
				if not abstract in response_dict[query]:
					if (len(response_dict.get(query)) < k): # parameter2
						response_dict[query].append(abstract)
					## these are listed in order, based on score
			else:
				response_dict[query] = [abstract]

	for query_id in range(1,total_queries):
		if (query_id in key_dict):
			total_answers =	len(key_dict[query_id])
		else:
			total_answers = 0
		if (query_id in key_dict) and (query_id in response_dict):
			correct = 0
			incorrect = 0
			so_far = 0
			milestone = .1
			precisions = []
			recordable_recall = 0

			for abstract_id in response_dict[query_id]:
				so_far = so_far+1
				if abstract_id in key_dict[query_id]:					
					correct = correct + 1
					recall = float(correct)/float(total_answers)
					if (correct+incorrect)<=total_answers:
						recordable_recall = recall
					while recall>milestone:
						precisions.append(float(correct)/float(correct+incorrect))
						milestone = milestone+.1
				else:					
					incorrect = incorrect+1
			if len(precisions)>0:
				average_precision = (sum(precisions))/len(precisions)
				all_precisions.append(average_precision)
				all_recalls.append(recordable_recall)
			else:
				missing_responses.append(query_id)
			# all_precisions.append(average_precision)
			# all_recalls.append(recordable_recall)
		elif query_id in key_dict:
			all_recalls.append(0)
	print (f'--------k:{k}-e:{e}--------')
	#print ('Queries with No responses:'+str(missing_responses))
	MAP = sum(all_precisions)/len(all_precisions)
	Recall = sum(all_recalls)/len(all_recalls)
	print ('Average Precision is: '+str(MAP))
	print ('Average Recall is: '+str(Recall))
	return MAP

def start_evaluate(e_list, k_list):
	print("==Evaluation==")
	dir = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/') + '/'
	key_file = dir + "./dataset/cranqrel.txt"
	response_file = dir + "./dataset/output.txt"
	maxK = 0
	maxE = 0
	maxPrecision = 0
	for e in e_list:
		for k in k_list:
			precision = evaluation(key_file,response_file, k, e)
			if (maxPrecision < precision):
				maxPrecision = precision
				maxK = k
				maxE = e

	print("==Best Parameter==")
	print(f"Precision: {maxPrecision} K: {maxK} E: {maxE}")