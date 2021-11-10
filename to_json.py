import pandas as pd
import json
df = pd.read_excel("data training.xlsx")
js_columns = df.to_json(orient = 'columns')
js_records = df.to_json(orient = 'records')
js_index = df.to_json(orient = 'index')
js_split = df.to_json(orient = 'split')


out = df.to_json(orient='table')[1:-1].replace('},{', '} {')
with open("table.json", "w") as outfile:
    json.dump(out, outfile)


def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

'''
data = pd.read_excel("data training.xlsx")
data = data.drop(columns=['sentiment'])

result = data.to_json(orient="values")
parsed = json.loads(result)
done = json.dumps(parsed, indent=4)'''
'''
dictionary = {
    "berita":jsonStr
}

# Serializing json
json_object = json.dumps(dictionary, indent=4)
print(json_object)
with open("sample.json", "w") as outfile:
    json.dump(dictionary, outfile)'''