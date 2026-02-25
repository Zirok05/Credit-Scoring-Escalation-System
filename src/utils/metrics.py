import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)
from hydra.core.hydra_config import HydraConfig
import pandas as pd
import numpy as np


def plot_roc_curve(y_test, test_pred, test_auc, task):
    """ROC кривая с Plotly"""
    fpr, tpr, _ = roc_curve(y_test, test_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Model (AUC = {test_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.5)',
        line=dict(color='gray', width=1, dash='dash')
    ))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800, height=600,
        showlegend=True,
        legend=dict(x=0.6, y=0.2)
    )

    task.get_logger().report_plotly("ROC Curve", f"test_auc_{test_auc:.3f}", figure=fig, iteration=0)


def plot_pr_curve(y_test, test_pred, task):
    """Precision-Recall кривая с Plotly"""
    precision, recall, _ = precision_recall_curve(y_test, test_pred)
    ap = average_precision_score(y_test, test_pred)
    baseline_ap = y_test.mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'Model (AP = {ap:.3f})',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[baseline_ap, baseline_ap],
        mode='lines',
        name=f'Random (AP = {baseline_ap:.3f})',
        line=dict(color='gray', width=1, dash='dash')
    ))

    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=800, height=600,
        showlegend=True
    )

    task.get_logger().report_plotly("PR Curve", "Default Class", figure=fig, iteration=0)
    return ap


def plot_confusion_matrix(y_test, test_pred, task, threshold=0.5):
    """Confusion Matrix с Plotly"""
    y_pred_class = (test_pred > threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_class)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Good', 'Predicted Default'],
        y=['Actual Good', 'Actual Default'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    ))

    fig.update_layout(
        width=500, height=500,
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )

    task.get_logger().report_plotly("Confusion Matrix", f"threshold={threshold}", figure=fig, iteration=0)


def plot_distribution(y_test, test_pred, task):
    """Распределение предсказаний с Plotly"""
    df_good = pd.DataFrame({'probability': test_pred[y_test == 0], 'class': 'Actual Good'})
    df_default = pd.DataFrame({'probability': test_pred[y_test == 1], 'class': 'Actual Default'})
    df_plot = pd.concat([df_good, df_default])

    fig = px.histogram(
        df_plot, x='probability', color='class',
        nbins=50, opacity=0.7,
        color_discrete_map={'Actual Good': 'green', 'Actual Default': 'red'},
        barmode='overlay'
    )

    fig.update_layout(
        xaxis_title='Predicted Probability of Default',
        yaxis_title='Count',
        width=1000, height=600,
        legend_title=''
    )

    task.get_logger().report_plotly("Distributions", "Predictions by True Class", figure=fig, iteration=0)


def plot_feature_importance(model, feature_names, task):
    """Feature importance для tree-based моделей с Plotly"""
    if hasattr(model, 'feature_importances_'):
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        model_name = model.__class__.__name__.replace('Classifier', '')

        # Таблица
        task.get_logger().report_table(
            "Feature Importance", model_name,
            table_plot=imp_df.sort_values('importance', ascending=False),
            iteration=0
        )

        # График
        fig = go.Figure(go.Bar(
            x=imp_df['importance'],
            y=imp_df['feature'],
            orientation='h',
            marker_color='steelblue'
        ))

        fig.update_layout(
            xaxis_title='Importance',
            yaxis_title='Feature',
            width=800, height=500
        )

        task.get_logger().report_plotly("Feature Importance Plot", model_name, figure=fig, iteration=0)


def plot_coefficients(model, feature_names, task):
    """Коэффициенты для линейных моделей с Plotly"""
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', ascending=True)

        # Таблица
        task.get_logger().report_table(
            "Feature Coefficients", "linear",
            table_plot=coef_df.sort_values('coefficient', ascending=False),
            iteration=0
        )

        # График с цветами
        colors = ['red' if x > 0 else 'green' for x in coef_df['coefficient']]

        fig = go.Figure(go.Bar(
            x=coef_df['coefficient'],
            y=coef_df['feature'],
            orientation='h',
            marker_color=colors
        ))

        fig.update_layout(
            #title='Top Feature Coefficients',
            xaxis_title='Coefficient',
            yaxis_title='Feature',
            width=800, height=500,
        )

        task.get_logger().report_plotly("Coefficients Plot", "Top Feature Coefficients", figure=fig, iteration=0)


def plot_LogisticRegressionCV(model, task, model_name="logreg"):
    """Логирование результатов LogisticRegressionCV"""

    # 1. Лучшие параметры
    best_C = model.C_[0]
    msg = f"Best C: {best_C:.4f}"

    # Проверяем, использовался ли elasticnet
    if hasattr(model, 'l1_ratios') and model.l1_ratios is not None:
        best_l1 = model.l1_ratio_[0]
        msg += f", best l1_ratio: {best_l1:.2f}"

    task.get_logger().report_text(msg)

    # 2. Таблица с результатами
    scores = model.scores_[1]

    # Определяем, был ли elasticnet по наличию l1_ratios и их не-None значению
    is_elasticnet = hasattr(model, 'l1_ratios') and model.l1_ratios is not None

    if is_elasticnet:
        # ElasticNet: scores имеет форму [n_folds, n_Cs, n_l1]
        mean_scores = scores.mean(axis=0)  # [n_Cs, n_l1]

        results = []
        for i, C in enumerate(model.Cs):
            for j, l1 in enumerate(model.l1_ratios):
                results.append({
                    'C': C,
                    'l1_ratio': l1,
                    'mean_test_score': mean_scores[i, j]
                })
        cv_results = pd.DataFrame(results).sort_values('mean_test_score', ascending=False)
    else:
        # L1/L2: scores имеет форму [n_folds, n_Cs]
        mean_scores = scores.mean(axis=0)
        cv_results = pd.DataFrame({
            'C': model.Cs,
            'mean_test_score': mean_scores
        }).sort_values('mean_test_score', ascending=False)

    task.get_logger().report_table(
        "CV Results",
        model_name,
        table_plot=cv_results.head(15),
        iteration=0
    )


def plot_GridSearchCV(grid_search, task, model_name="model"):
    """Логирование результатов GridSearchCV в ClearML"""

    # 1. Лучшие параметры
    task.get_logger().report_text(
        f"Best parameters for {model_name}: {grid_search.best_params_}, best score: {grid_search.best_score_}"
    )

    # 2. Таблица со всеми результатами
    cv_results = pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score', ascending=False)

    task.get_logger().report_table(
        "GridSearchCV Top 15 Results",
        model_name,
        table_plot=cv_results.head(15).fillna("None"),
        iteration=0
    )

    return cv_results


def plot_log_metrics(y_true, y_pred_proba, task, dataset_name="test", threshold=0.5):
    """Подсчет и логирование метрик классификации"""

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef
    )

    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        'ROC-AUC': roc_auc_score(y_true, y_pred_proba),
        'AP': average_precision_score(y_true, y_pred_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    # Логируем каждую метрику
    for name, value in metrics.items():
        task.get_logger().report_scalar(f"Metrics/{dataset_name}", name, value, 0)

    # Таблица с метриками
    metrics_df = pd.DataFrame([
        {'Metric': name, 'Value': f'{value:.4f}'}
        for name, value in metrics.items()
    ])

    task.get_logger().report_table(
        f"Metrics {dataset_name}",
        "summary",
        table_plot=metrics_df,
        iteration=0
    )

    return metrics