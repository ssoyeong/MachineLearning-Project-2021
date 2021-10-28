from LAB3_SearchEngine.infoRetrieval import *
import pandas as pd

def run():
  print("Process Query")
  qDocs = parseQuery("LAB3_SearchEngine/Cranfield_Collection/cran.qry")
  qToks = tokenize(qDocs, "query")
  qDic = organize(qToks, "query")
  # qIdf = idf(qDic, len(qDocs))
  # qTf = queryTf(qToks)


  print("Process Abstract Docs")
  absDocs = parseAbsDocs("LAB3_SearchEngine/Cranfield_Collection/cran.all.1400")
  absToks = tokenize(absDocs, "abstract")
  absDic = organize(absToks, "abstract")
  # absIdf = idf(absDic, len(absDocs))
  # absTf = abstractTf(absDic,len(absDocs))


  # Construct Inverted Index  TODO: CountVectorizer 안쓰고 tokenize() 구현되어 있는 거 바탕으로 했습니다.
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

  # print(invertedIndexDF)
  invertedIndexDF = invertedIndexDF[127:].reset_index()
  invertedIndexDF = invertedIndexDF.drop(['index'], axis=1)
  print(invertedIndexDF.head(30))


  '''
  print("Process Scoring" )
  scoreList = score(qToks, absTf, absIdf, qTf, qIdf)

  print("Saving the Result")
  createOutput("output.txt", scoreList)
  print("all done! results are saved in 'output.txt' file")
  '''

if __name__ == "__main__":
  run()
