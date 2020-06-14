from matplotlib.ticker import FormatStrFormatter

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
