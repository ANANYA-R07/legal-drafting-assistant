from app.corpus import load_corpus
from app.gpt2_classifier import FineTunedGPT2Classifier

corpus = load_corpus()
test_data = corpus["test"]
classifier = FineTunedGPT2Classifier()

print(f"Test samples count: {len(test_data)}")
correct = 0
for i, item in enumerate(test_data, 1):
    pred = classifier.predict(item["sentence"])
    is_corr = pred["predicted_category"] == item["clause_type"]
    if is_corr:
        correct += 1
    mark = "PASS" if is_corr else "FAIL"
    print(f"{i:2d}. [{mark}] True: {item['clause_type']:<15} | Pred: {pred['predicted_category']:<15} ({pred['confidence']*100:.1f}%) | Text: {item['sentence'][:55]}...")

acc = correct / len(test_data)
print(f"\nTotal correct: {correct}/{len(test_data)} ({acc*100:.2f}%)")
