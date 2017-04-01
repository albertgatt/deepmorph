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
vsub_features = ['person', 'number', 'gender', 'derived_form']
nsub_features = ['number', 'gender', 'form']

def get_verb_features(v, o):
	result = []

	if o in v and v[o] is not None:
		for s in vsub_features:
			if s in v[o]:
				result += [v[o][s]]
			else:
				result += [None]
	else:
		result += [None for x in vsub_features]

	return result

def get_noun_features(n):
	result = []
	nonetypes = ['sp', 'mf']
	for f in nsub_features:
		if f in n and n[f] is not None:
			value = n[f]

			if value in nonetypes:
				value = 'none'

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
			pos = lexeme['pos']

			if 'root' in lexeme and lexeme['root'] is not None:
				try:
					root = lexeme['root']['radicals']
				except KeyError:
					root = None


			for wf in db.wordforms.find({'lexeme_id': lexid, 'pending': {"$ne": True}}):
				i+=1

				if i % 1000 == 0:
					print(i)

				sf = wf['surface_form']

				#some words are multi-word expressions -- exclude
				if len(sf.split()) > 1:
					continue
				
				wf_features = [sf] #[lemma, root, sf] #initial feature vector
					
				if pos == 'VERB':
					#print(sf)
					try:
						wf_features += [wf['aspect'], wf['polarity']]
						wf_features += get_verb_features(wf, 'subject')
						#wf_features += get_verb_features(wf, 'dir_obj')
						#wf_features += get_verb_features(wf, 'ind_obj')
						data.write('\t'.join([str(x) for x in wf_features]) + "\n")
					except: #skip anything that doesn't have the expected features
						continue
				else:
					try:
						wf_features += get_noun_features(wf)
						data.write('\t'.join([str(x) for x in wf_features]) + "\n")
					except: #skip anything that doesn't have the expected features
						continue

		print("Found total " + str(i) + " wordforms")
		compress_file(gz_file, data_file)


def split(f, train, test, t=90):
	with open(f, 'r', encoding="utf-8") as data:
		cases = data.readlines()
		random.shuffle(cases)
	
		perc = int(0.9*len(cases))

		with open(train, 'w', encoding="utf-8") as training:
			training.writelines(cases[0:perc])

		with open(test, 'w', encoding="utf-8") as testing:
			testing.writelines(cases[perc:])


if __name__ == "__main__":
	#get_from_db(['VERB'], '../data/verbs.txt', '../data/gabra-verbs-all.tar.bz2')
	#get_from_db(['NOUN', 'ADJ'], '../data/noun-adj.txt', '../data/gabra-noun-adj-all.tar.bz2')
	#split('../data/verbs.txt', '../data/verbs-train.txt', '../data/verbs-test.txt')
	split('../data/noun-adj.txt', '../data/noun-adj-train.txt', '../data/noun-adj-test.txt')

		