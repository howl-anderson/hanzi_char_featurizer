from hanzi_char_featurizer import Featurizer

featurizer = Featurizer()

print("=== extract() ===")
result = featurizer.extract("明天")
print(result)

print("\n=== extract(as_numpy=True) ===")
result = featurizer.extract("明天", as_numpy=True)
print(result)

print("\n=== vocabulary ===")
print(featurizer.vocabulary)
