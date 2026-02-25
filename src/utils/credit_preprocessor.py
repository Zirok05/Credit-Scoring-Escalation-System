import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin


class CreditDataPreprocessor(BaseEstimator, TransformerMixin):

    # Полный препроцессинг данных

    def __init__(self,
                 NumberOfDependents_fill_value=0,
                 NumberOfDependents_up_threshold=10,
                 MonthlyIncome_fill_value=0,
                 RevolvingUtilizationOfUnsecuredLines_drop_threshold=2,
                 age_low_drop_threshold=18,
                 age_up_drop_threshold=80,
                 DebtRatio_up_threshold=5,
                 PastDueRiskScore_weights=[1.0, 1.2, 1.3],
                 NumberRealEstateLoansOrLines_drop_threshold=20,
                 drop_special_codes=False):
        self.NumberOfDependents_fill_value = NumberOfDependents_fill_value
        self.NumberOfDependents_up_threshold = NumberOfDependents_up_threshold

        self.MonthlyIncome_fill_value = MonthlyIncome_fill_value

        self.RevolvingUtilizationOfUnsecuredLines_drop_threshold = RevolvingUtilizationOfUnsecuredLines_drop_threshold

        self.age_low_drop_threshold = age_low_drop_threshold
        self.age_up_drop_threshold = age_up_drop_threshold

        self.DebtRatio_up_threshold = DebtRatio_up_threshold

        self.PastDueRiskScore_weights = PastDueRiskScore_weights

        self.NumberRealEstateLoansOrLines_drop_threshold = NumberRealEstateLoansOrLines_drop_threshold

        self.drop_special_codes = drop_special_codes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy['NumberOfDependents'] = X_copy['NumberOfDependents'].fillna(value=self.NumberOfDependents_fill_value)
        X_copy['NumberOfDependents'] = X_copy['NumberOfDependents'].clip(0, self.NumberOfDependents_up_threshold).copy()

        X_copy['MonthlyIncomeIsMissing'] = 0
        X_copy.loc[X_copy['MonthlyIncome'].isna(), 'MonthlyIncomeIsMissing'] = 1
        X_copy['MonthlyIncome'] = X_copy['MonthlyIncome'].fillna(value=self.MonthlyIncome_fill_value)

        X_copy['RevolvingUtilizationOverOne'] = 0.0
        X_copy.loc[X_copy['RevolvingUtilizationOfUnsecuredLines'] > 1, 'RevolvingUtilizationOverOne'] = 1.0
        X_copy['RevolvingUtilizationOfUnsecuredLines'] = X_copy['RevolvingUtilizationOfUnsecuredLines'].clip(0,
                                                                                                             1).copy()

        X_copy['DebtPayments'] = 0.0
        X_copy.loc[X_copy['MonthlyIncome'] == 0, 'DebtPayments'] = X_copy.loc[X_copy['MonthlyIncome'] == 0, 'DebtRatio']
        X_copy.loc[X_copy['MonthlyIncome'] != 0, 'DebtPayments'] = X_copy.loc[
                                                                       X_copy['MonthlyIncome'] != 0, 'DebtRatio'] * \
                                                                   X_copy.loc[
                                                                       X_copy['MonthlyIncome'] != 0, 'MonthlyIncome']
        X_copy['DebtRatio'] = X_copy['DebtRatio'].clip(0, self.DebtRatio_up_threshold).copy()

        X_copy['DebtPayments_over_10k'] = 0.0
        X_copy.loc[X_copy['DebtPayments'] > 10000, 'DebtPayments_over_10k'] = 1.0
        X_copy['DebtPayments'] = X_copy['DebtPayments'].clip(0, 10000).copy()

        X_copy['MonthlyIncome_over_20k'] = 0.0
        X_copy.loc[X_copy['MonthlyIncome'] >= 20000, 'MonthlyIncome_over_20k'] = 1.0
        X_copy['MonthlyIncome'] = X_copy['MonthlyIncome'].clip(0, 20000)

        X_copy['Code96'] = 0.0
        X_copy['Code98'] = 0.0
        X_copy.loc[X_copy['NumberOfTime30-59DaysPastDueNotWorse'] == 96, 'Code96'] = 1.0
        X_copy.loc[X_copy['NumberOfTime30-59DaysPastDueNotWorse'] == 98, 'Code98'] = 1.0

        X_copy['PastDueRiskScore'] = (
                self.PastDueRiskScore_weights[0] * X_copy['NumberOfTime30-59DaysPastDueNotWorse'] +
                self.PastDueRiskScore_weights[1] * X_copy['NumberOfTime60-89DaysPastDueNotWorse'] +
                self.PastDueRiskScore_weights[2] * X_copy['NumberOfTimes90DaysLate'])
        X_copy.loc[X_copy['NumberOfTime30-59DaysPastDueNotWorse'] == 96, 'PastDueRiskScore'] = 96
        X_copy.loc[X_copy['NumberOfTime30-59DaysPastDueNotWorse'] == 98, 'PastDueRiskScore'] = 98
        X_copy = X_copy.drop(columns=['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse',
                                      'NumberOfTimes90DaysLate'])

        X_copy['NumberOfOpenCreditLinesAndLoans_over_30'] = 0.0
        X_copy.loc[X_copy['NumberOfOpenCreditLinesAndLoans'] > 30, 'NumberOfOpenCreditLinesAndLoans_over_30'] = 1.0
        X_copy['NumberOfOpenCreditLinesAndLoans'] = X_copy['NumberOfOpenCreditLinesAndLoans'].clip(0, 30).copy()

        X_copy['NumberRealEstateLoansOrLines_over_5'] = 0.0
        X_copy.loc[X_copy['NumberRealEstateLoansOrLines'] > 5, 'NumberRealEstateLoansOrLines_over_5'] = 1.0
        X_copy['NumberRealEstateLoansOrLines'] = X_copy['NumberRealEstateLoansOrLines'].clip(0, 5).copy()

        X_copy['ConsumerCredit_Group'] = pd.cut(X_copy['NumberOfOpenCreditLinesAndLoans'],
                                                bins=[0, 1, 2, 6, 15, 31],
                                                labels=[
                                                    '0_loans',
                                                    '1_loans',
                                                    '2-5_loans',
                                                    '6-14_loans',
                                                    '16-30_loans'
                                                ])
        consumer_dummy = pd.get_dummies(X_copy['ConsumerCredit_Group'], prefix='Consumer', drop_first=False).astype(
            'float')

        X_copy['RealEstateLoans_Group'] = pd.cut(X_copy['NumberRealEstateLoansOrLines'],
                                                 bins=[-1, 0, 3, 100],
                                                 labels=[
                                                     '0_loans',
                                                     '1-3_loans',
                                                     '4+_loans',
                                                 ])
        estate_dummy = pd.get_dummies(X_copy['RealEstateLoans_Group'], prefix='RealEstateLoans',
                                      drop_first=False).astype('float')

        X_copy = pd.concat([X_copy, consumer_dummy, estate_dummy], axis=1).copy()
        X_copy = X_copy.drop(columns=['ConsumerCredit_Group',
                                      'RealEstateLoans_Group']).copy()

        X_copy = X_copy.drop(columns=['Consumer_6-14_loans',
                                      'RealEstateLoans_0_loans']).copy()

        X_copy = X_copy.drop(columns=['NumberOfOpenCreditLinesAndLoans',
                                      'NumberRealEstateLoansOrLines',
                                      'MonthlyIncomeIsMissing',
                                      'MonthlyIncome_over_20k',
                                      'Consumer_0_loans',
                                      'NumberOfOpenCreditLinesAndLoans_over_30']).copy()

        if self.drop_special_codes:
            X_copy = X_copy.drop(columns=['Code96', 'Code98'])

        return X_copy


    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def clean_train(self, X, y=None):
        mask = (
                (X[
                     'RevolvingUtilizationOfUnsecuredLines'] <= self.RevolvingUtilizationOfUnsecuredLines_drop_threshold) &
                (X['age'] >= self.age_low_drop_threshold) &
                (X['age'] <= self.age_up_drop_threshold) &
                (X['NumberRealEstateLoansOrLines'] <= self.NumberRealEstateLoansOrLines_drop_threshold)
        )

        X_clean = X[mask].copy()

        if y is not None:
            y_clean = y[mask].copy()
            return X_clean, y_clean

        return X_clean

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

class CreditScaler(BaseEstimator, TransformerMixin):
    """
    Масштабирует только не-булевые колонки.
    Можно задать различные способы масштабирования
    """

    def __init__(self, scaler_type='standard'):
        """
        Параметр scaler_type - тип scaler'а.

        Доступные типы:
        - 'standard': StandardScaler (среднее=0, дисперсия=1)
        - 'robust': RobustScaler (устойчив к выбросам)
        - 'minmax': MinMaxScaler (приводит к [0, 1])
        - 'maxabs': MaxAbsScaler (приводит к [-1, 1])
        """

        self.boolean_columns = [
            'RevolvingUtilizationOverOne',
            'DebtPayments_over_10k',
            'Code96',
            'Code98',
            'NumberRealEstateLoansOrLines_over_5',
            'Consumer_1_loans',
            'Consumer_2-5_loans',
            'Consumer_16-30_loans',
            'RealEstateLoans_1-3_loans',
            'RealEstateLoans_4+_loans'
        ]

        self.scaler_type = scaler_type
        self._create_scaler()

        # Эти переменные заполнятся во время fit
        self.columns_to_scale_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None

    def _create_scaler(self):
        """Создает scaler по типу"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'maxabs':
            self.scaler = MaxAbsScaler()
        else:
            raise ValueError(
                f"Unknown scaler_type: {self.scaler_type}. "
                f"Available: standard, robust, minmax, maxabs"
            )

    def fit(self, X, y=None):
        """
        Определяет колонки для масштабирования (все, кроме булевых)
        и обучает scaler.
        """

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        self.columns_to_scale_ = [
            col for col in self.feature_names_in_
            if col not in self.boolean_columns
        ]

        self.scaler.fit(X[self.columns_to_scale_])
        return self

    def transform(self, X, y=None):
        """
        Масштабирует только не-булевы колонки.
        """
        X_copy = X.copy()

        X_copy[self.columns_to_scale_] = self.scaler.transform(X_copy[self.columns_to_scale_])

        return X_copy

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def get_feature_names_out(self, input_features=None):
        """Для совместимости с sklearn"""
        if input_features is not None:
            return input_features
        return self.feature_names_in_ if self.feature_names_in_ is not None else []

    def set_params(self, **params):
        """Для совместимости с GridSearchCV"""
        if 'scaler_type' in params and params['scaler_type'] != self.scaler_type:
            self.scaler_type = params['scaler_type']
            self._create_scaler()
        return super().set_params(**params)


def check_business_rules(age, monthly_income, monthly_debt, debt_ratio,
                         late_90, late_60_89, late_30_59, credit_lines,
                         real_estate, utilization, dependents):

    # КРИТИЧЕСКИЕ ПРАВИЛА - сразу отказ
    if age < 18:
        return {
            'needs_manual': False,
            'message': 'Возраст менее 18 лет - кредит не выдаётся',
            'decision': 1  # отказ
        }

    # СПЕЦИАЛЬНЫЕ БАНКОВСКИЕ КОДЫ - сразу ручной разбор
    if (late_90 == 98) or (late_60_89 == 98) or (late_30_59 == 98):
        return {
            'needs_manual': True,
            'message': 'Код 98: Списание долга как безнадежного',
            'decision': None
        }

    if (late_90 == 96) or (late_60_89 == 96) or (late_30_59 == 96):
        return {
            'needs_manual': True,
            'message': 'Код 96: Изъятие залога или реализация имущества',
            'decision': None
        }

    # КРИТИЧЕСКИЕ ПРАВИЛА - сразу ручной разбор
    if age > 80:
        return {
            'needs_manual': True,
            'message': 'Возраст > 80 лет - требуется ручной разбор (индивидуальные условия)',
            'decision': None
        }

    if monthly_income > 1000000:
        return {
            'needs_manual': True,
            'message': 'Доход свыше 1,000,000 $ - требуется ручной разбор',
            'decision': None
        }

    if monthly_debt > 1000000:
        return {
            'needs_manual': True,
            'message': 'Платежи свыше 1,000,000 $ - требуется ручной разбор',
            'decision': None
        }

    if utilization > 2:
        return {
            'needs_manual': True,
            'message': 'Использование кредитных средств превышает 200%',
            'decision': None
        }

    if real_estate > 20:
        return {
            'needs_manual': True,
            'message': 'Количество кредитов под залог недвижимости слишком велико - ручной разбор',
            'decision': None
        }

    # 4. ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ - допуск к авторазбору моделью
    return {
        'needs_manual': False,
        'decision': None,
    }



