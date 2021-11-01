from processing import *
from evaluating import *
from searching import *
import pandas as pd

def createOutput(fileName, scores):
  with open(fileName, "w") as f:
    for score in scores:
      for tup in score:
        string = ""
        for t in tup:
          string = string + str(t) + " "
        f.write(string + "\n")
    f.close()

def constructInvertedIndex(absDic):
  # Construct Inverted Index
  termList = list(absDic.keys())
  docList = []
  docValues = list(absDic.values())
  for doc in docValues:
    matchDoc = []
    for x in range(len(doc)):
      if doc[x] > 0:
        matchDoc.append(x+1)
    docList.extend([matchDoc])

  invertedIndex = dict(zip(termList, docList))
  invertedIndex = dict(sorted(invertedIndex.items()))
  invertedIndexDF = pd.DataFrame({'Term':list(invertedIndex.keys()),
                                 'Document':list(invertedIndex.values())})

  invertedIndexDF = invertedIndexDF[127:].reset_index()
  invertedIndexDF = invertedIndexDF.drop(['index'], axis=1)
  # print(invertedIndexDF.head(30))
  invertedIndexDF.to_csv("dataset/invertedIndex.csv")
  return invertedIndexDF

def processing():
  # Docmument preprocessing
  print("==Process Abstract Docs==")
  absDocs = parseAbsDocs("dataset/cran.all.1400.txt")
  absToks = tokenize(absDocs, "abstract")
  absDic = organize(absToks, "abstract")
  absIdf = idf(absDic, len(absDocs))
  absTf = abstractTf(absDic,len(absDocs))

  # Construct Inverted Index
  print("==Process costruct inverted index==")
  invertedIndex = constructInvertedIndex(absDic)

  # Query preprocessing
  print("==Process Query==")
  qDocs = parseQuery("dataset/cran.qry.txt")
  qToks = tokenize(qDocs, "query")
  qDic = organize(qToks, "query")
  qIdf = idf(qDic, len(qDocs))
  qTf = queryTf(qToks)

  # Scoring
  print("===Process Scoring===" )
  scoreList = make_score_list(qToks, absTf, absIdf, qTf, qIdf)

  # Make Output
  print("===Saving the Result===")
  createOutput("dataset/output.txt", scoreList)
  print("Saving File Complete!")

if __name__ == "__main__":
  # processing()

  ##### Query-based Search Engine #####
  e = [0.3, 0.4, 0.5, 0.6, 0.7]
  k = [3, 5, 10, 30, 50, 100]
  # start_evaluate(e, k)

  ##### Word-based Search Engine #####
  input = ['affected', 'afterbody']
  # Search for documents that match the input words
  output = start_search_word(input)
  if len(output) == 0:
    output = 'No matching results.'
  print("Search word:", input)
  print("Matching document:", output)
