import json

# example dictionary to save as JSON
data = {
    "first_name": "John",
    "last_name": "Doe",
    "email": "john@doe.com",
    "salary": 1499.9, # just to demonstrate we can use floats as well
    "age": 17,
    "is_real": False, # also booleans!
    "titles": ["The Unknown", "Anonymous"] # also lists!
}

# save JSON file
# 1st option
with open("data1.json", "w") as f:
    json.dump(data, f)

# 2nd option
with open("data2.json", "w") as f:
    f.write(json.dumps(data, indent=4))


unicode_data = {
    "first_name": "أحمد",
    "last_name": "علي"
}

with open("data_unicode.json", "w", encoding="utf-8") as f:
    json.dump(unicode_data, f, ensure_ascii=False)