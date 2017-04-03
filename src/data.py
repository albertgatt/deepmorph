import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import tarfile
import random

client = MongoClient('localhost', 27017)
db = client.gabra
verb_file = '../data/verbs.txt'
other_file = '../data/nouns.txt'

#
vfeatures = ['aspect', 'polarity']
vsub_features = ['person', 'number', 'gender']
nsub_features = ['number', 'gender', 'form']

def get_verb_features(v, o):
	result = []

	if o in v and v[o] is not None:
		for s in vsub_features:
			if s in v[o]:
				result += [v[o][s]]
			elif s == 'gender':
				result += ['mf']
			else:
				raise RuntimeError('Not found: ' + s)
	else:
		result += [None for x in vsub_features]

	return result

def get_noun_features(n):
	result = []
	mapping = {'sgv': 'sg', 'verbalnoun': 'none', 'mimated': 'none', 'pl_ind': 'pl'}
	

	for f in nsub_features:
		if f in n:
			value = n[f]

			if value is None or len(value) == 0:
				value = 'none'
			elif n[f] in mapping:
				value = mapping[value]

			result += [value]
		
		else:
			result += ['none']

	return result


def compress_file(outfile, fname):
	tar = tarfile.open(outfile + ".tar.bz2", "w:bz2")
	tar.add(fname)
	tar.close()

def get_from_db(pos, data_file, gz_file):
	with open(data_file, 'w', encoding="utf-8") as data:
		i = 0;
		for lexeme in db.lexemes.find({'pos': {"$in": pos}}):
			
			lexid = ObjectId(lexeme['_id'])
			lemma = lexeme['lemma']
			root = None
			vform = 'NF'
			pos = lexeme['pos']

			if 'root' in lexeme and lexeme['root'] is not None:
				try:
					root = lexeme['root']['radicals']
				except KeyError:
					root = None

			if 'derived_form' in lexeme:				
				vform = lexeme['derived_form']

			for wf in db.wordforms.find({'lexeme_id': lexid, 'pending': {"$ne": True}}):
				i+=1

				if i % 1000 == 0:
					print(i)

				sf = wf['surface_form'].strip().lower()

				#exclude non-words, words with a hyphen or multiword expressions
				if len(sf) < 1 or '-' in sf or len(sf.split()) > 1:
					continue
				
				wf_features = [] #[lemma, root, sf] #initial feature vector
					
				if pos == 'VERB':
					#map 'imperative to a mood, not aspect'
					try:
						mood = 'ind'							
						aspect = wf['aspect']

						if aspect == 'imp':
							mood = 'imp'
							aspect = 'impf'						

						wf_features += [vform, aspect, mood, wf['polarity']]
						wf_features += get_verb_features(wf, 'subject')
						#wf_features += get_verb_features(wf, 'dir_obj')
						#wf_features += get_verb_features(wf, 'ind_obj')
						data.write(sf + "\t" + ' - '.join([str(x) for x in wf_features]) + "\n")
					except: #skip anything that doesn't have the expected features
						continue
				else:
					try:						
						wf_features += get_noun_features(wf)
						data.write(sf + '\t' + ' - '.join([str(x) for x in wf_features]) + "\n")					

					except: #skip anything that doesn't have the expected features
						continue

		print("Found total " + str(i) + " wordforms")
		compress_file(gz_file, data_file)

def reformat(line):
	items = line.split("\t")
	result = items[0] + "\t" + " - ".join(items[1:])
	return result


def split(f, train, test, t=90, reformat=False):
	with open(f, 'r', encoding="utf-8") as data:
		cases = data.readlines()
		random.shuffle(cases)
	
		perc = int(0.9*len(cases))

		with open(train, 'w', encoding="utf-8") as training:
			if reformat:
				training.writelines(map(reformat, cases[0:perc]))
			else:
				training.writelines(cases[0:perc])

			compress_file(train + ".tar.bz2", train)

		with open(test, 'w', encoding="utf-8") as testing:
			if reformat:
				testing.writelines(map(reformat, cases[perc:]))
			else:
				testing.writelines(cases[perc:])
				
			compress_file(test + ".tar.bz2", test)


if __name__ == "__main__":
	get_from_db(['VERB'], 'verbs-mood-form-all.txt', 'gabra-verbs-mood-form-all')
	#get_from_db(['NOUN', 'ADJ'], 'noun-adj.txt', 'gabra-noun-adj-all')
	split('verbs-mood-form-all.txt', 'gabra-verbs-mood-form-train.txt', 'gabra-verbs-mood-form-test.txt')
	#split('noun-adj.txt', 'gabra-noun-adj-train', 'gabra-noun-adj-test')

		