# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-27

class TaskType:
    Classification = "classification"
    MultilabelClassification = "multilabel_classification"

    @classmethod
    def str(cls):
        return ",".join([cls.Classification, cls.MultilabelClassification])


class ModeType:
    """Standard names for util modes.
    The following standard keys are defined:
    * `TRAIN`: training mode.
    * `EVAL`: evaluation mode.
    * `PREDICT`: inference mode.
    """
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'

    @classmethod
    def str(cls):
        return ",".join([cls.TRAIN, cls.EVAL, cls.PREDICT])

    @classmethod
    def lists(cls):
        return [cls.TRAIN, cls.EVAL, cls.PREDICT]


# def get_optimal_threshold(y_true, y_pred, label_list, eval_func, independent=True, lower_better=False,
#                           **eval_fn_kwargs):
#     """
#         Performs grid search on best performing thresholds from list
#         > [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
#         using eval_func.
#
#         Input:
#             - y_true : DataFrame of one-hot encoded true labels of shape (n_samples, n_labels)
#             - y_pred : DataFrame predicted logits of shape (n_samples, n_labels)
#             - label_list : list of labels (same order as label columns for y_true and y_pred)
#             - eval_func : metric to optimize thresholds against
#             - independent: if True, optimizes threshold per label independently. Else, globally
#             - lower_better : If lower values are better for eval_func, set lower_better=True
#             - eval_fn_kwargs : Extra arguments to be used in eval_func. Example: when optimizing
#                                thresholds globally, you would want to pass 'average'=
#
#         Example usage:
#            > import src.model.serve as serve
#            > from sklearn import metrics
#            > logits_train = serve.predict(df_train, model, device, label_list)
#            > best_thr, best_scores = get_optimal_threshold(train_df, logits_train,
#                                                           label_list,
#                                                           metrics.f1_score,
#                                                           independent=True,
#                                                           lower_better=False,
#                                                         #   **{'average':'micro'}
#                                                           )
#
#     """
#     assert isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame)
#     assert y_true.shape[0] == y_pred.shape[0]
#
#     thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
#
#     if independent:
#         result = {}
#         result_eval = {}
#         for label in label_list:
#             truth = y_true[label]
#             preds = y_pred[label]
#
#             best_score = 0
#             best_thr = 0
#             for thr in thresholds:
#                 hard_preds = np.array(preds > thr).astype(int)
#                 score = eval_func(truth, hard_preds, **eval_fn_kwargs)
#
#                 if (lower_better and score < best_score) or (
#                         not lower_better and score > best_score):
#                     best_score = score
#                     best_thr = thr
#
#             result[label] = best_thr
#             result_eval[label] = best_score
#
#     else:
#         print(eval_fn_kwargs)
#         best_score = 0
#         best_thr = 0
#         for thr in thresholds:
#             y_pred_discrete = logits_to_discrete(y_pred, label_list, thr)
#             score = eval_func(y_true[label_list], y_pred_discrete[label_list], **eval_fn_kwargs)
#
#             if (lower_better and score < best_score) or (not lower_better and score > best_score):
#                 best_score = score
#                 best_thr = thr
#         result = best_thr
#         result_eval = best_score
#
#     return result, result_eval
