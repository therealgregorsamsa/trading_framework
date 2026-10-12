"""Walk-forward accuracy of the baseline and the two models."""

from hffx.dataset import load_dataset
from hffx.models import baseline_model, boosting_model, evaluate_model, logistic_model


def main() -> None:
    df = load_dataset("EURUSD")
    for name, model in [
        ("baseline", baseline_model()),
        ("logistic", logistic_model()),
        ("boosting", boosting_model()),
    ]:
        scores = evaluate_model(df, model)
        print(name, round(sum(scores) / len(scores), 4))


if __name__ == "__main__":
    main()
