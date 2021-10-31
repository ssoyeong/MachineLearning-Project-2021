from infoRetrieval import *
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
  print(invertedIndexDF.head(30))
  return invertedIndexDF

def run():
  # Docmument preprocessing
  print("==Process Abstract Docs==")
  absDocs = parseAbsDocs("LAB3_SearchEngine/Cranfield_Collection/cran.all.1400.txt")
  absToks = tokenize(absDocs, "abstract")
  absDic = organize(absToks, "abstract")
  absIdf = idf(absDic, len(absDocs))
  absTf = abstractTf(absDic,len(absDocs))

  # Construct Inverted Index
  print("==Process costruct inverted index==")
  invertedIndex = constructInvertedIndex(absDic)

  # Query preprocessing
  print("==Process Query==")
  qDocs = parseQuery("LAB3_SearchEngine/Cranfield_Collection/cran.qry.txt")
  qToks = tokenize(qDocs, "query")
  qDic = organize(qToks, "query")
  qIdf = idf(qDic, len(qDocs))
  qTf = queryTf(qToks)

  # Evaluation
  print("===Process Scoring===" )
  scoreList = score(qToks, absTf, absIdf, qTf, qIdf)

  # Output
  print("===Saving the Result===")
  createOutput("LAB3_SearchEngine/output.txt", scoreList)
  print("All done!")

if __name__ == "__main__":
  run()
