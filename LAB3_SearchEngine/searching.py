import pandas as pd


def start_search_word(searchWord):
    print("==Word-based Search Engine==")
    df = pd.read_csv("dataset/invertedIndex.csv", header=0, index_col=0)
    docCol = df.loc[:]['Document']

    # Convert 'Document' column from string to int
    idx = 0
    for doc in docCol:
        doc = doc[1:-1]
        doc = list(map(int, doc.split(', ')))
        df.loc[idx]['Document'] = doc
        idx += 1

    # Binary search
    terms = df[:]['Term']
    resultDocs = set()
    isFirst = 1
    for t in searchWord:
        x = binary_search(t, terms)
        if x == None:
            break;
        if isFirst:
            resultDocs.update(df.loc[x][1])
            isFirst = 0
        resultDocs = resultDocs.intersection(df.loc[x][1])

    return sorted(resultDocs)


def binary_search(target, data):
    data.sort_values()
    start = 0
    end = len(data) - 1

    while start <= end:
        mid = (start + end) // 2

        if data[mid] == target:
            return mid
        elif data[mid] < target:
            start = mid + 1
        else:
            end = mid - 1

    return None
