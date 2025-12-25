import csv
import os
import re
from datetime import datetime

base_path = os.getcwd()

print("Введите названия входных файлов:")
dataset_filename = input("Имя файла с данными (например, c-103-1.txt): ").strip()
coords_filename = input("Имя файла с координатами (например, Координаты С-103.txt): ").strip()

dataset_file = os.path.join(base_path, dataset_filename)
coords_file = os.path.join(base_path, coords_filename)

output_file1 = os.path.join(base_path, "data_80_no.csv")
output_file2 = os.path.join(base_path, "data_20_no.csv")
output_file3 = os.path.join(base_path, "data_80_noise.csv")
output_file4 = os.path.join(base_path, "data_20_noise.csv")


def time_to_ms(hhmmss: str, msec_str: str) -> float:
    try:
        t = datetime.strptime(hhmmss, "%H:%M:%S")
        msec = float(msec_str.replace(",", "."))
        return (t.hour * 3600 + t.minute * 60 + t.second) * 1000 + msec
    except:
        return None


# координаты сенсоров
coords = {}
with open(coords_file, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
            except:
                pass


# загружаем все строки
lines = []
with open(dataset_file, encoding="utf-8") as f:
    for line in f:
        l = line.strip()
        if l and not l.startswith(("La", "Dt")):
            lines.append(l)

groups = []      
noise_lines = []    
current_group = []

for line in lines:

    if line.startswith("LE"):
        # завершаем старую группу
        if current_group:
            groups.append(current_group)
        # начинаем новую
        current_group = [line]

    elif line.startswith("Ht"):
        if current_group:
            current_group.append(line)
        else:
            # Ht без LE — считаем шумом
            noise_lines.append(line)

    elif line.startswith("Ev"):
        if current_group:
            noise_lines.append(line)
            groups.append(current_group)
            current_group = []
        else:
            noise_lines.append(line)


# добавляем последнюю группу
if current_group:
    groups.append(current_group)


total_groups = len(groups)
n80 = int(total_groups * 0.8)

groups_80 = groups[:n80]
groups_20 = groups[n80:]

# шум также делим 80/20
n_noise80 = int(len(noise_lines) * 0.8)
noise_80 = noise_lines[:n_noise80]
noise_20 = noise_lines[n_noise80:]

def write_dataset(groups_list, noise_list, output_clean, output_noise):

    with open(output_clean, "w", newline="", encoding="utf-8") as fc, \
         open(output_noise, "w", newline="", encoding="utf-8") as fn:

        wc = csv.writer(fc)
        wn = csv.writer(fn)

        header = ["DSET", "LABLE", "Time_ms", "Sensor", "X", "Y", "Group"]
        wc.writerow(header)
        wn.writerow(header)

        # --- Чистые данные (только LE/Ht) ---
        group_id = 0
        for group in groups_list:
            group_id += 1

            for line in group:
                parts = re.split(r"\s+", line.replace(",", "."))
                if len(parts) < 5:
                    continue

                dset, id_, hhmmss, msec_str = parts[0], parts[1], parts[2], parts[3]
                sensor = int(parts[4]) if parts[4].isdigit() else -1

                total_ms = time_to_ms(hhmmss, msec_str)
                x, y = coords.get(sensor, ("", ""))

                wc.writerow([id_, dset, total_ms, sensor, x, y, group_id])
                wn.writerow([id_, dset, total_ms, sensor, x, y, group_id])

        # --- Шум ---
        for line in noise_list:
            parts = re.split(r"\s+", line.replace(",", "."))
            if len(parts) < 4:
                continue

            dset = parts[0]
            id_ = parts[1]
            hhmmss = parts[2]
            msec_str = parts[3]
            sensor = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else -1

            total_ms = time_to_ms(hhmmss, msec_str)
            x, y = coords.get(sensor, ("", ""))

            wn.writerow([id_, dset, total_ms, sensor, x, y, -1])


write_dataset(groups_80, noise_80, output_file1, output_file3)
write_dataset(groups_20, noise_20, output_file2, output_file4)

print("Готово!")
print(output_file1)
print(output_file2)
print(output_file3)
print(output_file4)



import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns




def build_spatial_patterns(train_data):
    """
    средний центр (X,Y)
    средняя дистанция до центра
    стандартное отклонение расстояний до центра
    ковариация (X,Y)
    """

    patterns = {}

    groups = sorted(train_data["Group"].unique())

    for g in groups:
        df = train_data[train_data["Group"] == g]

        center = np.array([df["X"].mean(), df["Y"].mean()])
        coords = df[["X", "Y"]].values
        dists = np.linalg.norm(coords - center, axis=1)

        pattern = {
            "center": center,
            "mean_dist": dists.mean(),
            "std_dist": dists.std() if len(dists) > 1 else 0,
            "cov": np.cov(coords.T) if len(coords) > 2 else np.eye(2)
        }

        patterns[g] = pattern

    return patterns



# Пост-обработка шума


def spatial_noise_postprocessing(test_data, test_clusters, spatial_patterns):
    """
    Переводит неверно распознанные кластеры в шум,
    сравнивая форму и разброс кластера с обученными паттернами.
    """

    updated = test_clusters.copy()

    # Уникальные кластеры кроме шума
    pred_clusters = [c for c in np.unique(test_clusters) if c != -1]

    for cl in pred_clusters:
        # Выбираем точки текущего кластера
        df = test_data[test_clusters == cl]
        # Вычисляем пространственные характеристики кластера

        coords = df[["X", "Y"]].values
        center = coords.mean(axis=0)
        dists = np.linalg.norm(coords - center, axis=1)

        mean_dist = dists.mean()
        std_dist = dists.std() if len(dists) > 1 else 0

        # Поиск ближайшего обученного паттерна по расстоянию между центрами
        best_match = None
        smallest_center_shift = 999999

        for g, p in spatial_patterns.items():
            shift = np.linalg.norm(center - p["center"])
            if shift < smallest_center_shift:
                smallest_center_shift = shift
                best_match = g

        pattern = spatial_patterns[best_match]

        #  Критерии похожести кластера на обученный
        valid = True
        # Центр кластера не должен быть слишком далеко от центра паттерна
        if smallest_center_shift > pattern["mean_dist"] * 2:
            valid = False
        # Cреднее расстояние не должно быть слишком большим
        if mean_dist > pattern["mean_dist"] * 1.0:
            valid = False
         # Разброс точек не должен быть слишком большим
        if std_dist > pattern["std_dist"] * 1.0:
            valid = False
        # Минимальное количество точек в кластере
        if len(df) < 3:
            valid = False

        # Если кластер не проходит проверки → шум
        if not valid:
            updated[test_clusters == cl] = -1

    return updated
    
def print_overall_spatial_stats(spatial_patterns):
    """
    Сводные статистики по пространственным паттернам.
    """

    all_mean_dists = []
    all_std_dists = []

    print("\n ОБЩАЯ СТАТИСТИКА ОБУЧАЮЩИХ ДАННЫХ")

    for g, p in spatial_patterns.items():

        if g == -1:     # <-- игнорируем шум
            continue

        all_mean_dists.append(p["mean_dist"])
        all_std_dists.append(p["std_dist"])

    #Общие значения
    print(f"Средняя mean_dist по всем КЛАССАМ: {np.mean(all_mean_dists):.2f}")
    print(f"Средняя std_dist  по всем КЛАССАМ: {np.mean(all_std_dists):.2f}")

    #Подробно
    print("\nПодробно по каждому классу:")

    for g, p in spatial_patterns.items():

        if g == -1:
            continue

        print(f"Группа {g}: mean_dist = {p['mean_dist']:.2f}, std_dist = {p['std_dist']:.2f}")
        
def print_spatial_patterns(patterns):
    print("\n=== ПРОСТРАНСТВЕННЫЕ ПАТТЕРНЫ ОБУЧАЮЩИХ ДАННЫХ ===")
    for g, p in patterns.items():
        print(f"\nГруппа {g}:")
        print(f"  Центр: ({p['center'][0]:.2f}, {p['center'][1]:.2f})")
        print(f"  Средняя дистанция: {p['mean_dist']:.2f}")
        print(f"  Std дистанции: {p['std_dist']:.2f}")
        print(f"  Ковариация:\n{p['cov']}")


def simple_clustering_approach(train_data, test_data):
    """
    Кластеризация на основе Time_ms
    """

    basic_features = ['Time_ms']

    X_train_basic = train_data[basic_features].values
    X_test_basic = test_data[basic_features].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_basic)
    X_test_scaled = scaler.transform(X_test_basic)

    best_ari = -1
    best_params = {}

    eps_range = np.linspace(0.00001, 0.001, 100)
    min_samples_range = range(2, 5)

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            train_pred = dbscan.fit_predict(X_train_scaled)

            n_clusters = len(np.unique(train_pred)) - (1 if -1 in train_pred else 0)
            if n_clusters < 2:
                continue

            ari = adjusted_rand_score(train_data['Group'], train_pred)

            if ari > best_ari:
                best_ari = ari
                best_params = {'eps': eps, 'min_samples': min_samples}

    print(f"Лучшие параметры: eps={best_params['eps']:.6f}, min_samples={best_params['min_samples']}")
    print(f"ARI на обучающих данных: {best_ari:.4f}")

    # Объединяем данные
    X_combined = np.vstack([X_train_scaled, X_test_scaled])
    dbscan_final = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    combined_clusters = dbscan_final.fit_predict(X_combined)

    # Кластеры "до постобработки"
    test_clusters_before_refine = combined_clusters[len(train_data):]

    #ВЫВОД КЛАСТЕРОВ ДО ПОСТОБРАБОТКИ
    print("\n=== Кластеры после DBSCAN (по времени), до постобработки ===")
    unique_pre = np.unique(test_clusters_before_refine)
    print(f"Количество кластеров (без учёта шума): {len(unique_pre) - (1 if -1 in unique_pre else 0)}")
    print(f"Шумовых точек: {np.sum(test_clusters_before_refine == -1)}")

    for cid in unique_pre:
        count = np.sum(test_clusters_before_refine == cid)
        print(f"  Кластер {cid}: {count} точек")

    # Пост-обработка шума по пространственным паттернам
    spatial_patterns = build_spatial_patterns(train_data)
    test_clusters_refined = spatial_noise_postprocessing(test_data, test_clusters_before_refine, spatial_patterns)

    #ВЫВОД КЛАСТЕРОВ ПОСЛЕ ПОСТОБРАБОТКИ
    print("\n=== Кластеры после постобработки (пространственные признаки) ===")
    unique_post = np.unique(test_clusters_refined)
    print(f"Количество кластеров (без учёта шума): {len(unique_post) - (1 if -1 in unique_post else 0)}")
    print(f"Шумовых точек: {np.sum(test_clusters_refined == -1)}")

    for cid in unique_post:
        count = np.sum(test_clusters_refined == cid)
        print(f"  Кластер {cid}: {count} точек")

    return test_clusters_before_refine, test_clusters_refined, best_params

def visualize_test_results(test_data, clusters_before, clusters_after):
    """
    Создает комплексную визуализацию результатов кластеризации ТЕСТОВЫХ данных.
    
    Создает 4 графика:
    1. Пространственное распределение тестовых данных до постобработки
    2. Пространственное распределение тестовых данных после постобработки
    3. Временное распределение кластеров до постобработки
    4. Временное распределение кластеров после постобработки
    """
    plt.figure(figsize=(15, 12))
    
    # 1. Тестовые данные до постобработки (пространство)
    plt.subplot(2, 2, 1)
    unique_clusters_before = np.unique(clusters_before)
    
    # Создаем цветовую палитру для всех кластеров
    n_clusters_before = len(unique_clusters_before)
    colors_before = plt.cm.tab20(np.linspace(0, 1, max(n_clusters_before, 1)))
    
    for i, cluster in enumerate(unique_clusters_before):
        mask = clusters_before == cluster
        if np.sum(mask) > 0:  # Проверяем, что есть точки в кластере
            label = 'Шум' if cluster == -1 else f'Кластер {cluster}'
            plt.scatter(test_data.loc[mask, 'X'], test_data.loc[mask, 'Y'],
                       label=label, color=colors_before[i], alpha=0.7, s=60,
                       edgecolors='black', linewidth=0.5)
    
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.title('Тестовые данные: пространственное распределение\n(ДО постобработки)')
    if n_clusters_before <= 10:  # Добавляем легенду только если кластеров не слишком много
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 2. Тестовые данные после постобработки (пространство)
    plt.subplot(2, 2, 2)
    unique_clusters_after = np.unique(clusters_after)
    
    # Создаем цветовую палитру для всех кластеров
    n_clusters_after = len(unique_clusters_after)
    colors_after = plt.cm.tab20(np.linspace(0, 1, max(n_clusters_after, 1)))
    
    for i, cluster in enumerate(unique_clusters_after):
        mask = clusters_after == cluster
        if np.sum(mask) > 0:  # Проверяем, что есть точки в кластере
            label = 'Шум' if cluster == -1 else f'Кластер {cluster}'
            plt.scatter(test_data.loc[mask, 'X'], test_data.loc[mask, 'Y'],
                       label=label, color=colors_after[i], alpha=0.7, s=60,
                       edgecolors='black', linewidth=0.5)
    
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.title('Тестовые данные: пространственное распределение\n(ПОСЛЕ постобработки)')
    if n_clusters_after <= 10:  # Добавляем легенду только если кластеров не слишком много
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 3. Временное распределение кластеров до постобработки
    plt.subplot(2, 2, 3)
    
    # Группируем данные по кластерам для гистограммы
    time_data = test_data['Time_ms'].values
    cluster_labels_before = clusters_before
    
    # Определяем диапазон времени для всех гистограмм
    time_min, time_max = time_data.min(), time_data.max()
    time_bins = 30
    
    for i, cluster in enumerate(unique_clusters_before):
        mask = cluster_labels_before == cluster
        if np.sum(mask) > 0:
            label = 'Шум' if cluster == -1 else f'Кл {cluster}'
            cluster_times = time_data[mask]
            
            # Используем alpha для наложения гистограмм
            plt.hist(cluster_times, bins=time_bins, 
                    alpha=0.5, label=label, color=colors_before[i],
                    range=(time_min, time_max), density=True)
    
    plt.xlabel('Время (мс)')
    plt.ylabel('Плотность распределения')
    plt.title('Временное распределение кластеров\n(ДО постобработки)')
    if n_clusters_before <= 8:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 4. Временное распределение кластеров после постобработки
    plt.subplot(2, 2, 4)
    
    cluster_labels_after = clusters_after
    
    for i, cluster in enumerate(unique_clusters_after):
        mask = cluster_labels_after == cluster
        if np.sum(mask) > 0:
            label = 'Шум' if cluster == -1 else f'Кл {cluster}'
            cluster_times = time_data[mask]
            
            plt.hist(cluster_times, bins=time_bins, 
                    alpha=0.5, label=label, color=colors_after[i],
                    range=(time_min, time_max), density=True)
    
    plt.xlabel('Время (мс)')
    plt.ylabel('Плотность распределения')
    plt.title('Временное распределение кластеров\n(ПОСЛЕ постобработки)')
    if n_clusters_after <= 8:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Дополнительная визуализация: сравнение количества точек в кластерах
    plt.figure(figsize=(12, 5))
    
    # Подготовка данных для сравнения
    counts_before = []
    labels_before = []
    
    # Сортируем кластеры по убыванию количества точек
    cluster_counts_before = []
    for cluster in unique_clusters_before:
        count = np.sum(clusters_before == cluster)
        cluster_counts_before.append((cluster, count))
    
    # Сортируем по количеству точек
    cluster_counts_before.sort(key=lambda x: x[1], reverse=True)
    
    for cluster, count in cluster_counts_before:
        labels_before.append('Шум' if cluster == -1 else f'Кл {cluster}')
        counts_before.append(count)
    
    # Аналогично для after
    counts_after = []
    labels_after = []
    
    cluster_counts_after = []
    for cluster in unique_clusters_after:
        count = np.sum(clusters_after == cluster)
        cluster_counts_after.append((cluster, count))
    
    cluster_counts_after.sort(key=lambda x: x[1], reverse=True)
    
    for cluster, count in cluster_counts_after:
        labels_after.append('Шум' if cluster == -1 else f'Кл {cluster}')
        counts_after.append(count)
    
    # Создаем индексы для графиков
    x_before = np.arange(len(counts_before))
    x_after = np.arange(len(counts_after))
    
    # Два отдельных графика для лучшей читаемости
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x_before, counts_before, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Кластеры')
    plt.ylabel('Количество точек')
    plt.title('Размеры кластеров ДО постобработки')
    plt.xticks(x_before, labels_before, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения над столбцами
    for i, v in enumerate(counts_before):
        plt.text(i, v + max(counts_before)*0.01, str(v), ha='center', fontsize=9)
    
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x_after, counts_after, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Кластеры')
    plt.ylabel('Количество точек')
    plt.title('Размеры кластеров ПОСЛЕ постобработки')
    plt.xticks(x_after, labels_after, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения над столбцами
    for i, v in enumerate(counts_after):
        plt.text(i, v + max(counts_after)*0.01, str(v), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('test_cluster_sizes_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Визуализация центров кластеров и их перемещения
    if 'Group' in test_data.columns:
        visualize_cluster_movement(test_data, clusters_before, clusters_after)

def visualize_cluster_movement(test_data, clusters_before, clusters_after):
    """
    Визуализирует изменение центров кластеров после постобработки.
    """
    plt.figure(figsize=(10, 8))
    
    # Вычисляем центры кластеров до постобработки
    centers_before = {}
    unique_before = np.unique(clusters_before)
    
    for cluster in unique_before:
        if cluster != -1:  # Пропускаем шум
            mask = clusters_before == cluster
            if np.sum(mask) >= 3:  # Только для кластеров с достаточным количеством точек
                center_x = test_data.loc[mask, 'X'].mean()
                center_y = test_data.loc[mask, 'Y'].mean()
                centers_before[cluster] = (center_x, center_y, np.sum(mask))
    
    # Вычисляем центры кластеров после постобработки
    centers_after = {}
    unique_after = np.unique(clusters_after)
    
    for cluster in unique_after:
        if cluster != -1:  # Пропускаем шум
            mask = clusters_after == cluster
            if np.sum(mask) >= 3:  # Только для кластеров с достаточным количеством точек
                center_x = test_data.loc[mask, 'X'].mean()
                center_y = test_data.loc[mask, 'Y'].mean()
                centers_after[cluster] = (center_x, center_y, np.sum(mask))
    
    # Рисуем центры до постобработки
    for cluster, (x, y, size) in centers_before.items():
        plt.scatter(x, y, s=size*2, color='blue', alpha=0.6, 
                   label='До постобработки' if cluster == list(centers_before.keys())[0] else '',
                   edgecolors='darkblue', linewidth=1.5)
        plt.text(x, y, f' {cluster}', fontsize=9, color='darkblue', fontweight='bold')
    
    # Рисуем центры после постобработки
    for cluster, (x, y, size) in centers_after.items():
        plt.scatter(x, y, s=size*2, color='red', alpha=0.6, 
                   label='После постобработки' if cluster == list(centers_after.keys())[0] else '',
                   edgecolors='darkred', linewidth=1.5, marker='s')
        plt.text(x, y, f' {cluster}', fontsize=9, color='darkred', fontweight='bold')
    
    # Соединяем центры стрелками (если кластер сохранился)
    for cluster_before in centers_before:
        for cluster_after in centers_after:
            # Упрощенная логика соединения (можно улучшить)
            x1, y1, _ = centers_before[cluster_before]
            x2, y2, _ = centers_after[cluster_after]
            
            # Проверяем близость центров
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < 5:  # Порог близости
                plt.arrow(x1, y1, x2-x1, y2-y1, 
                         head_width=0.3, head_length=0.5, 
                         fc='gray', ec='gray', alpha=0.5,
                         length_includes_head=True)
    
    plt.xlabel('X координата')
    plt.ylabel('Y координата')
    plt.title('Изменение центров кластеров после постобработки')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_cluster_centers_movement.png', dpi=300)
    plt.show()
def main():

    train_data = pd.read_csv("data_80_noise.csv")
    test_data = pd.read_csv("data_20_noise.csv")

    print(f"Обучающие данные: {len(train_data)} сигналов, {len(train_data['Group'].unique())} групп")
    print(f"Тестовые данные: {len(test_data)} сигналов")

    clusters_before, clusters_after, params = simple_clustering_approach(train_data, test_data)
    spatial_patterns = build_spatial_patterns(train_data)
    print_overall_spatial_stats(spatial_patterns)

    # Визуализация результатов ТЕСТОВЫХ данных
    print("\nГенерация визуализаций ТЕСТОВЫХ данных")
    visualize_test_results(test_data, clusters_before, clusters_after)

    #ДОБАВЛЯЕМ В CSV
    test_data['Cluster_Time'] = clusters_before     # до постобработки
    test_data['Cluster_Final'] = clusters_after     # после постобработки
    test_data['Predicted_Group_Label'] = ['Noise' if x == -1 else f'Cluster_{x}' for x in clusters_after]

    if 'Group' in test_data.columns:
        test_ari = adjusted_rand_score(test_data['Group'], clusters_after)
        print(f"\nARI на тестовых данных: {test_ari:.4f}")

    # сохраняем csv
    test_data.to_csv('test_data_with_clusters.csv', index=False)
    params_df = pd.DataFrame([params])
    params_df.to_csv('dbscan_model_params.csv', index=False)

    print("\nФайл test_data_with_clusters.csv сохранён:")
    print("Cluster_Time  — кластеры по времени (до постобработки)")
    print("Cluster_Final — кластеры после пространственной постобработки")
    print("Predicted_Group_Label — метка кластера")


if __name__ == "__main__":
    main()
