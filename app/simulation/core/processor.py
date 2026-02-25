import numpy as np
import pandas as pd
from app.models.escalation import escalation_decision
from app.models.escalation import check_business_rules
def processing_time_function(lr_proba, second_proba, threshold=0.5, base_time=5,
                             lr_weight=1.0, second_weight=1.5):
    """
    Генерирует время обработки для заявок, попавших в ручной разбор
    """
    total_weight = lr_weight + second_weight
    proba = (lr_proba * lr_weight + second_proba * second_weight) / total_weight

    margin = abs(proba - threshold)
    max_margin = max(threshold, 1 - threshold)
    uncertainty = 1 - (margin / max_margin)

    mean_time = base_time * (1 + 3 * uncertainty)
    processing_time = np.random.exponential(scale=mean_time)

    return max(1, processing_time)


class ApplicationProcessor:
    def __init__(self, lr_model, second_model, second_model_name,
                 specialists_count=5,  # основные специалисты (модели)
                 business_specialists_count=2,  # эксперты (бизнес-правила)
                 base_processing_time=5, # время рассмотрения анкет от моделей
                 business_processing_time=10,  # время рассмотрения анкет от бизнес-правил
                 lr_weight=1.0, second_weight=1.5):
        self.lr_model = lr_model
        self.second_model = second_model
        self.second_model_name = second_model_name
        self.specialists_count = specialists_count
        self.business_specialists_count = business_specialists_count
        self.base_processing_time = base_processing_time
        self.business_processing_time = business_processing_time
        self.lr_weight = lr_weight
        self.second_weight = second_weight

        self.specialists = [0] * specialists_count
        self.business_specialists = [0] * business_specialists_count  # отдельный пул
        self.manual_queue = []  # очередь от моделей
        self.business_queue = []  # очередь от бизнес-правил

        self.stats = {
            'total_processed': 0,
            'auto_approved': 0,
            'auto_declined': 0,
            'manual_sent': 0,
            'manual_processed': 0,
            'business_manual_sent': 0,
            'business_manual_processed': 0,
            'queue_history': [],
            'business_queue_history': [],
            'wait_times': [],
            'business_wait_times': [],
            'specialist_busy': [],
            'business_specialist_busy': [],
            'business_rules_manual': 0,
            'business_rules_auto': 0
        }
        self.batch_stats = []

    def process_batch(self, applications_batch, preprocessor, scaler,
                      threshold, lr_margins, second_margins, current_time):
        """
        Обрабатывает батч заявок за текущую минуту
        """
        minute_results = {
            'new_apps': len(applications_batch),
            'auto_decisions': [],
            'new_manual': 0,
            'new_business_manual': 0,
            'processed_manual': 0,
            'processed_business_manual': 0,
            'queue_size': 0,
            'business_queue_size': 0,
            'specialists_busy': sum(1 for s in self.specialists if s > 0),
            'business_specialists_busy': sum(1 for s in self.business_specialists if s > 0),
            'business_rules': 0
        }

        # 1. Уменьшаем время работы специалистов
        self.specialists = [max(0, s - 1) for s in self.specialists]
        self.business_specialists = [max(0, s - 1) for s in self.business_specialists]

        if not applications_batch:
            minute_results['queue_size'] = len(self.manual_queue)
            minute_results['business_queue_size'] = len(self.business_queue)
            self.stats['queue_history'].append(len(self.manual_queue))
            self.stats['business_queue_history'].append(len(self.business_queue))
            self.stats['specialist_busy'].append(minute_results['specialists_busy'])
            self.stats['business_specialist_busy'].append(minute_results['business_specialists_busy'])
            return minute_results

        df = pd.DataFrame(applications_batch)

        # 3. Применяем бизнес-правила ко всем заявкам
        manual_mask, auto_reject_mask, messages, auto_decisions = check_business_rules(df)

        # Сохраняем статистику по бизнес-правилам
        business_manual_count = manual_mask.sum()
        business_auto_count = auto_reject_mask.sum()

        # Инициализируем
        n = len(applications_batch)
        model_indices = []

        # 4. Обрабатываем результаты бизнес-правил
        for idx in range(n):
            if manual_mask[idx]:
                # Ручной разбор по бизнес-правилам - в отдельную очередь
                self.business_queue.append({
                    'app': applications_batch[idx],
                    'arrival_time': current_time,
                    'reason': 'business_rules',
                    'message': messages[idx],
                    'lr_proba': None,
                    'second_proba': None
                })
                minute_results['new_business_manual'] += 1
                minute_results['business_rules'] += 1
                self.stats['business_rules_manual'] += 1
                self.stats['business_manual_sent'] += 1

            elif auto_reject_mask[idx]:
                # Автоматический отказ по бизнес-правилам
                decision = {
                    'final_decision': auto_decisions[idx],
                    'model_used': 'Business Rules',
                    'probability': 1.0,
                    'needs_review': False,
                    'message': messages[idx]
                }
                minute_results['auto_decisions'].append(decision)
                self.stats['auto_declined'] += 1
                self.stats['business_rules_auto'] += 1
                self.stats['total_processed'] += 1

            else:
                # Заявка идет в модели
                model_indices.append(idx)

        # Инициализируем переменные для статистики моделей
        lr_confident_count = 0
        second_confident_count = 0
        second_uncertain_count = 0

        # 5. Батчевая обработка моделей
        if model_indices:
            # Берём только заявки, которые прошли бизнес-правила
            df_models = df.iloc[model_indices].copy()

            # Формируем DataFrame для моделей
            model_df = pd.DataFrame({
                'RevolvingUtilizationOfUnsecuredLines': df_models['RevolvingUtilizationOfUnsecuredLines'],
                'age': df_models['age'],
                'NumberOfTime30-59DaysPastDueNotWorse': df_models['NumberOfTime30-59DaysPastDueNotWorse'],
                'DebtRatio': df_models['DebtRatio'].fillna(0),
                'MonthlyIncome': df_models['MonthlyIncome'].fillna(0),
                'NumberOfOpenCreditLinesAndLoans': df_models['NumberOfOpenCreditLinesAndLoans'],
                'NumberOfTimes90DaysLate': df_models['NumberOfTimes90DaysLate'],
                'NumberRealEstateLoansOrLines': df_models['NumberRealEstateLoansOrLines'],
                'NumberOfTime60-89DaysPastDueNotWorse': df_models['NumberOfTime60-89DaysPastDueNotWorse'],
                'NumberOfDependents': df_models['NumberOfDependents'].fillna(0)
            })

            # Вызываем escalation_decision для всего батча
            batch_decisions, batch_manual_mask, stats = escalation_decision(
                model_df,
                self.lr_model,
                self.second_model,
                self.second_model_name,
                threshold=threshold,
                lr_margins=lr_margins,
                second_margins=second_margins,
                preprocessor=preprocessor,
                scaler=scaler
            )

            # Сохраняем статистику из escalation_decision
            lr_confident_count = stats['lr_confident']
            second_confident_count = stats['second_confident']
            second_uncertain_count = stats['second_uncertain']

            # Распределяем результаты по исходным индексам
            for local_idx, orig_idx in enumerate(model_indices):
                decision = batch_decisions[local_idx]

                if decision['needs_review']:
                    self.manual_queue.append({
                        'app': applications_batch[orig_idx],
                        'arrival_time': current_time,
                        'reason': 'model_uncertainty',
                        'decision': decision,
                        'lr_proba': decision.get('lr_proba'),
                        'second_proba': decision.get('second_proba')
                    })
                    minute_results['new_manual'] += 1
                    self.stats['manual_sent'] += 1
                else:
                    minute_results['auto_decisions'].append(decision)
                    if decision['final_decision'] == 0:
                        self.stats['auto_approved'] += 1
                    else:
                        self.stats['auto_declined'] += 1

                self.stats['total_processed'] += 1

        # Сохраняем общую статистику батча
        self.batch_stats.append({
            'time': current_time,
            'business_manual': business_manual_count,
            'business_auto': business_auto_count,
            'lr_confident': lr_confident_count,
            'second_confident': second_confident_count,
            'second_uncertain': second_uncertain_count,
            'total_in_batch': len(applications_batch),
            'new_manual': minute_results['new_manual'],
            'new_business_manual': minute_results['new_business_manual'],
            'auto_total': len(minute_results['auto_decisions'])
        })

        # 6. Распределяем заявки из бизнес-очереди по свободным экспертам
        for i in range(self.business_specialists_count):
            if self.business_specialists[i] <= 0 and self.business_queue:
                next_app = self.business_queue.pop(0)

                wait_time = current_time - next_app['arrival_time']
                self.stats['business_wait_times'].append(wait_time)

                # Эксперты обрабатывают бизнес-правила
                proc_time = self.business_processing_time

                self.business_specialists[i] = proc_time
                minute_results['processed_business_manual'] += 1
                self.stats['business_manual_processed'] += 1

        # 7. Распределяем заявки из основной очереди по свободным специалистам
        for i in range(self.specialists_count):
            if self.specialists[i] <= 0 and self.manual_queue:
                next_app = self.manual_queue.pop(0)

                wait_time = current_time - next_app['arrival_time']
                self.stats['wait_times'].append(wait_time)

                if next_app['reason'] == 'business_rules':
                    proc_time = self.business_processing_time
                else:
                    proc_time = processing_time_function(
                        lr_proba=next_app.get('lr_proba', 0.5),
                        second_proba=next_app.get('second_proba', 0.5),
                        threshold=threshold,
                        base_time=self.base_processing_time,
                        lr_weight=self.lr_weight,
                        second_weight=self.second_weight
                    )

                self.specialists[i] = proc_time
                minute_results['processed_manual'] += 1
                self.stats['manual_processed'] += 1

        minute_results['queue_size'] = len(self.manual_queue)
        minute_results['business_queue_size'] = len(self.business_queue)
        self.stats['queue_history'].append(len(self.manual_queue))
        self.stats['business_queue_history'].append(len(self.business_queue))
        self.stats['specialist_busy'].append(minute_results['specialists_busy'])
        self.stats['business_specialist_busy'].append(minute_results['business_specialists_busy'])

        return minute_results

    def load_test_dataset(self, filepath):
        df = pd.read_csv(filepath)
        if 'SeriousDlqin2yrs' in df.columns:
            df = df.drop(columns=['SeriousDlqin2yrs'])
        return df.to_dict('records')

    def get_queue_stats(self):
        if self.stats['wait_times']:
            avg_wait = np.mean(self.stats['wait_times'])
            max_wait = np.max(self.stats['wait_times'])
        else:
            avg_wait = max_wait = 0

        if self.stats['business_wait_times']:
            avg_business_wait = np.mean(self.stats['business_wait_times'])
            max_business_wait = np.max(self.stats['business_wait_times'])
        else:
            avg_business_wait = max_business_wait = 0

        return {
            'current_queue': len(self.manual_queue),
            'current_business_queue': len(self.business_queue),
            'avg_wait_minutes': avg_wait,
            'max_wait_minutes': max_wait,
            'avg_business_wait_minutes': avg_business_wait,
            'max_business_wait_minutes': max_business_wait,
            'queue_history': self.stats['queue_history'],
            'business_queue_history': self.stats['business_queue_history'],
            'specialist_busy': self.stats['specialist_busy'],
            'business_specialist_busy': self.stats['business_specialist_busy'],
            'business_rules_split': {
                'manual': self.stats['business_rules_manual'],
                'auto': self.stats['business_rules_auto']
            }
        }
