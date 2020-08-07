from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import fractions



def draw_prc_roc(y_test, y_score, recall_threshold, fpr_threshold):
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    fpr, tpr, thr = roc_curve(y_test, y_score, )
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))

    ax[0].step(recall, precision, color='b', alpha=0.2,
               where='post')
    ax[0].fill_between(recall, precision, step='post', alpha=0.2,
                       color='b')
    ax[0].grid(True)
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_yticks(np.arange(0, 1.05, 0.05))
    ax[0].set_xticks(np.arange(0, 1.05, 0.05))
    ax[0].set_xticklabels(np.arange(0, 1.05, 0.05), rotation='vertical')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    lw = 2
    ax[1].plot(fpr, tpr, color='darkorange')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].grid(True)
    ax[1].set_yticks(np.arange(0, 1.05, 0.05))
    ax[1].set_xticks(np.arange(0, 1.05, 0.05))
    ax[1].set_xticklabels(np.arange(0, 1.05, 0.05), rotation='vertical')
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver operating characteristic')
    ax[1].legend(loc="lower right")

    plt.show()


def positive_sampler(questions, answers):
    if len(questions) != len(answers):
        raise ValueError('Length of array of questions must be equal to length of array of answers.')
    for q, a in zip(questions, answers):
        yield q, a, 1

# старая версия
# def negative_sampler(question, questions, answers, max_tries=10000):
#     ## Будем генерировать случайные айдишники, если question == questions[случайный_айдишник], то семплируем пока это условие не
#     # примет значение False.
#     random_ids = iter(np.random.randint(0, len(questions), 1000000))
#     counter = 0
#     while True:
#         try:
#             rand_id = next(random_ids)
#         except StopIteration:
#             random_ids = iter(np.random.randint(0, len(questions), 1000000))
#             rand_id = next(random_ids)
#         ## если случайно вытянули айдишник того-же самого вопроса, пробуем ещё раз, пока случайный айдишник не будет относиться к
#         # другому вопросу.
#         random_question, random_answer = questions[rand_id], answers[rand_id]
#         if question != random_question:
#             yield question, random_answer, 0
#             counter = 0
#         else:
#             counter += 1
#             if counter >= max_tries:
#                 raise ValueError(
#                     'Could not sample negative example after {} tries. More unique questions must be added'.format(
#                         counter))
#             continue
# новая версия
def negative_sampler(question, questions, answers, max_tries=10000, n_rand_samp=1000000):
    random_ids = iter(np.random.randint(0, len(questions), n_rand_samp))
    counter = 0
    while True:
        try:
            rand_id = next(random_ids)
        except StopIteration:
            random_ids = iter(np.random.randint(0, len(questions), n_rand_samp))
            rand_id = next(random_ids)
        random_answer, random_question = answers[rand_id], questions[rand_id]
        if tuple(question) != tuple(random_question):
            counter = 0
            yield question, random_answer, 0
        else:
            counter += 1
            if counter >= max_tries:
                raise ValueError('Could not sample negative example after {} tries. More unique questions must be added'.format(
                                 counter))

def sampler(questions, answers, pos_frac, random_seed, max_tries=10000):
    if pos_frac > 0.5:
        raise NotImplementedError('Only < 0.5 pos_frac values can be used currently.')
    ##TODO: исправить тесты после закомменчивания проверок. Комментим, потому что раньше принимали стринги
    # а сейчас массивы с индексами. Теперь не нужно каждый раз тратить время на энкодинг
    if len(questions) != len(answers):
        raise ValueError('Length of array of questons must be equal to length of array of answers.')
    np.random.seed(random_seed)
    ## pos_frac = 0.3 -> 3/10 -> 1/3.(3) -> ~ 1/3 => на 3 семпла будет 1 позитивная и
    # 2 негативных пары
    pos_frac = fractions.Fraction(pos_frac).limit_denominator()
    nnegatives = round(pos_frac.denominator / pos_frac.numerator) - 1
    positives = positive_sampler(questions, answers)
    for question in questions:
        negatives = negative_sampler(question, questions, answers, max_tries)
        yield next(positives)
        for _ in range(nnegatives):
            yield next(negatives)

def draw_prc_roc(y_test, y_score, recall_threshold, fpr_threshold):
    from matplotlib.ticker import FormatStrFormatter
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    fpr, tpr, thr = roc_curve(y_test, y_score, )
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))

    ax[0].step(recall, precision, color='b', alpha=0.2,
               where='post')
    ax[0].fill_between(recall, precision, step='post', alpha=0.2,
                       color='b')
    ax[0].grid(True)
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_yticks(np.arange(0, 1.05, 0.05))
    ax[0].set_xticks(np.arange(0, 1.05, 0.05))
    ax[0].set_xticklabels(np.arange(0, 1.05, 0.05), rotation='vertical')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    lw = 2
    ax[1].plot(fpr, tpr, color='darkorange')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].grid(True)
    ax[1].set_yticks(np.arange(0, 1.05, 0.05))
    ax[1].set_xticks(np.arange(0, 1.05, 0.05))
    ax[1].set_xticklabels(np.arange(0, 1.05, 0.05), rotation='vertical')
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver operating characteristic')
    ax[1].legend(loc="lower right")

    plt.show()
    print('Class balance (pos/neg): ', round(np.mean(y_test), 3), ' / ', round(1 - np.mean(y_test), 3))
    print('precision(recall = {}) = '.format(recall_threshold),
          round(float(precision[::-1][np.searchsorted(recall[::-1], recall_threshold)]), 3))
    print('TPR on FPR({}) = '.format(fpr_threshold), round(tpr[(np.abs(fpr - fpr_threshold)).argmin()], 3))
    print('ROC-AUC: ', roc_auc_score(y_test, y_score))
    return fpr, tpr, thr

def save_pickle(obj, fpath):
    import pickle
    with open(fpath, 'wb') as fout:
        pickle.dump(obj, fout)
    return

def load_pickle(fpath):
    import pickle
    with open(fpath, 'rb') as fin:
        res = pickle.load(fin)
    return res