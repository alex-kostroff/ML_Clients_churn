import os


os.chdir(os.path.relpath("C:/Users/alex-/OneDrive/Desktop/aclImdb/"))

import os
import random

def load_training_data(
    data_directory: str = "C:/Users/alex-/OneDrive/Desktop/aclImdb/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Загрузка данных из файлов
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}", encoding='utf-8') as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label}
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]

import os
import random
import spacy

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Строим конвейер
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)

import os
import random
import spacy
from spacy.util import minibatch, compounding

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Строим конвейер
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Обучаем только textcat
    training_excluded_pipes = [                               #
        pipe for pipe in nlp.pipe_names if pipe != "textcat"  #
    ]                                                         #
    with nlp.disable_pipes(training_excluded_pipes):          #
        optimizer = nlp.begin_training()                      #
        # Итерация обучения
        print("Начинаем обучение")                            #
        batch_sizes = compounding(                            #
            4.0, 32.0, 1.001                                  #
        )  # Генератор бесконечной последовательности входных чисел
        for i in range(iterations):  #
            loss = {}  #
            random.shuffle(training_data)  #
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:  #
                text, labels = zip(*batch)  #
                nlp.update(  #
                    text,  #
                    labels,  #
                    drop=0.2,  #
                    sgd=optimizer,  #
                    losses=loss  #
                )  #

def evaluate_model(tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    # Указываем TP как малое число, чтобы в знаменателе
    # не оказался 0
    TP, FP, TN, FN = 1e-8, 0, 0, 0
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        score_pos = review.cats['pos']
        if true_label['pos']:
            if score_pos >= 0.5:
                TP += 1
            else:
                FN += 1
        else:
            if score_pos >= 0.5:
                FP += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


import os
import random
import spacy
from spacy.util import minibatch, compounding


def train_model(
        training_data: list,
        test_data: list,
        iterations: int = 20) -> None:
    # Строим конвейер
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Обучаем только textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Начинаем обучение")
        print("Loss\t\tPrec.\tRec.\tF-score")  #
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # Генератор бесконечной последовательности входных чисел
        for i in range(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text,
                    labels,
                    drop=0.2,
                    sgd=optimizer,
                    losses=loss
                )
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(  #
                    tokenizer=nlp.tokenizer,  #
                    textcat=textcat,  #
                    test_data=test_data  #
                )  #
                print(f"{loss['textcat']:9.6f}\t\
{evaluation_results['precision']:.3f}\t\
{evaluation_results['recall']:.3f}\t\
{evaluation_results['f-score']:.3f}")

    # Сохраняем модель                                 #
    with nlp.use_params(optimizer.averages):  #
        nlp.to_disk("model_artifacts")  #

train, test = load_training_data(limit=5000)
train_model(train, test, iterations=10)

def test_model(input_data: str):
    # Загружаем сохраненную модель
    loaded_model = spacy.load("model_artifacts")
    parsed_text = loaded_model(input_data)
    # Определяем возвращаемое предсказание
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный отзыв"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Негативный отзыв"
        score = parsed_text.cats["neg"]
    print(f"Текст обзора: {input_data}\n\
Предсказание: {prediction}\n\
Score: {score:.3f}")