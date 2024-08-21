# Adaptation from https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py


def tag2ot(ote_tag_sequence):
    """
    transform ote tag sequence to a sequence of opinion target
    :param ote_tag_sequence: tag sequence for ote task
    :return: A list of tuples where each tuple represents the start and end indices of an opinion target.
    """
    n_tags = len(ote_tag_sequence)
    ot_sequence = []
    beg, end = -1, -1

    for i in range(n_tags):
        # 'S' indicates a single word opinion target
        # 'B' indicates the beginning of a multi-word opinion target
        # 'E' indicates the end of a multi-word opinion target

        tag = ote_tag_sequence[i]
        if tag == 'S':
            ot_sequence.append((i, i))
        elif tag == 'B':
            beg = i
        elif tag == 'E':
            end = i
            if end > beg > -1: # Ensure valid beginning and end indices
                ot_sequence.append((beg, end))
                beg, end = -1, -1
    return ot_sequence


def evaluate_ote(gold_ot, pred_ot):
    """
    evaluate the model performance for the ote task
    :param gold_ot: gold standard ote tags
    :param pred_ot: predicted ote tags
    :return: A tuple containing precision, recall, and F1-score.
    """

    SMALL_POSITIVE_CONST = 1e-4

    assert len(gold_ot) == len(pred_ot)
    n_samples = len(gold_ot)

    # Initialize counters for true positives, gold standard, and predicted opinion targets
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    for i in range(n_samples):
        g_ot = gold_ot[i]
        p_ot = pred_ot[i]

        # Convert tag sequences to opinion target sequences
        g_ot_sequence, p_ot_sequence = tag2ot(ote_tag_sequence=g_ot), tag2ot(ote_tag_sequence=p_ot)
        # Count the number of correctly predicted opinion targets
        n_hit_ot = match_ot(gold_ote_sequence=g_ot_sequence, pred_ote_sequence=p_ot_sequence)

        n_tp_ot += n_hit_ot
        n_gold_ot += len(g_ot_sequence)
        n_pred_ot += len(p_ot_sequence)

    # calculate precision, recall and f1 for ote task
    ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)

    ote_scores = (ot_precision, ot_recall, ot_f1)
    return ote_scores


def evaluate(gold_ot, pred_ot):
    """
    evaluate the performance of the predictions
    :param gold_ot: gold standard opinion target tags
    :param pred_ot: predicted opinion target tags
    :return: metric scores of ner and sa
    """
    assert len(gold_ot) ==  len(pred_ot)
    ote_scores = evaluate_ote(gold_ot=gold_ot, pred_ot=pred_ot)
    return ote_scores


def match_ot(gold_ote_sequence, pred_ote_sequence):
    """
    calculate the number of correctly predicted opinion target
    :param gold_ote_sequence: gold standard opinion target sequence
    :param pred_ote_sequence: predicted opinion target sequence
    :return: Number of correctly predicted opinion targets.
    """
    n_hit = 0
    for t in pred_ote_sequence:
        if t in gold_ote_sequence:
            n_hit += 1
    return n_hit
