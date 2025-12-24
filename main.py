from src.utils import load_data, accuracy_score, confusion_metrics
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.logistic import LogisticRegression
from src.models.neural_net import NeuralNetwork
import sys
import pandas as pd

def main():
    print("[1/3] loading data...")
    try:
        X_train, X_test, y_train, y_test = load_data('data/captcha.csv')
    except FileNotFoundError:
        print("error: csv not found")
        sys.exit(1)

    print(f"   samples: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # config
    models = {
        "Naive Bayes": GaussianNaiveBayes(),
        "Logistic Regression": LogisticRegression(learning_rate=0.1, n_iters=10000),
        "Neural Network": NeuralNetwork(input_size=X_train.shape[1], hidden_size=16, epochs=5000)
    }

    results = []
    print("\n[2/3] benchmark start...")
    
    for name, model in models.items():
        print(f"   -> training: {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec = confusion_metrics(y_test, y_pred)
        
        results.append({
            "model": name,
            "accuracy": f"{acc:.2%}",
            "precision": f"{prec:.2%}",
            "recall": f"{rec:.2%}"
        })

    print("\n[3/3] results")
    print(pd.DataFrame(results).to_markdown(index=False))

if __name__ == "__main__":
    main()