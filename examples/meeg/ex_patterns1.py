# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import umap
import scipy.io
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh

# %%
import sys
# sys.path.append("C:/Users/ansbel/Documents/GitHub/site-packages/umap_meeg")
from topological_spatial_filter import fit_filters, fit_patterns
import torch

# %%
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"

raw = mne.io.read_raw_fif(fpath,preload=True)
sfreq = raw.info['sfreq']

# %%
new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
]

descriptions = raw.annotations.description

# Индексы значимых аннотаций (не BAD, не EDGE)
significant_mask = np.array([('BAD' not in d and 'EDGE' not in d) for d in descriptions])
significant_indices = np.where(significant_mask)[0]

# Проверка соответствия
if len(significant_indices) != len(new_descriptions):
    raise ValueError(
        f"Несоответствие: {len(significant_indices)} значимых, ожидается {len(new_descriptions)}"
    )

# Создаём новый список описаний
new_desc = list(descriptions)
for idx, label in zip(significant_indices, new_descriptions):
    new_desc[idx] = label

# Создаём новый объект Annotations с обновлёнными описаниями
old_annot = raw.annotations
new_annot = mne.Annotations(
    onset=old_annot.onset,
    duration=old_annot.duration,
    description=np.array(new_desc, dtype='U20'),  # тип с запасом по длине
    orig_time=old_annot.orig_time
)

# Применяем новые аннотации к raw_ica
raw.set_annotations(new_annot)

# Проверяем изменение
print(raw.annotations.description)

# %% 
# Получаем данные после ICA (все каналы EEG)
raw_clean = raw.copy().pick_types(eeg=True)
data = raw.get_data()  

# Списки для хранения отфильтрованных и обрезанных кусков
signal_pieces = []
noise_pieces = []
times_pieces = []

# Параметры фильтров
b_signal, a_signal = butter(3, np.array([15, 25]) / (int(sfreq) / 2), btype='band')
b_broad, a_broad = butter(3, np.array([13, 30]) / (int(sfreq) / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 25.5]) / (int(sfreq) / 2), btype='stop')

# Длительность для обрезания краёв (в секундах)
crop_duration = 0.5  # 0.5 секунды с каждой стороны
crop_samples = int(crop_duration * sfreq)

# Проходим по аннотациям
for annot in raw.annotations:
    desc = annot['description']
    if desc == 'BAD_':
    # if desc == 'BAD_' or desc == 'RS_EC_1' or desc == 'RS_EC_2':
        continue  
    duration = annot['duration']
    if duration < 2.0:  # слишком короткий блок - пропускаем
        continue
    print(desc)
    
    tmin = annot['onset']
    tmax = tmin + duration
    if tmax > raw.times[-1]:
        tmax = raw.times[-1]
    
    # Вырезаем кусок данных
    start_idx = int(tmin * sfreq)
    end_idx = int(tmax * sfreq)
    seg_data = data[:, start_idx:end_idx]
    
    # Фильтрация сигнала (15–25 Гц)
    seg_signal = filtfilt(b_signal, a_signal, seg_data, axis=1)
    
    # Фильтрация шума: широкополосный 13–27 Гц, затем режекция
    seg_noise_broad = filtfilt(b_broad, a_broad, seg_data, axis=1)
    seg_noise = filtfilt(b_stop, a_stop, seg_noise_broad, axis=1)
    
    # Обрезаем края для удаления переходных процессов
    if seg_signal.shape[1] > 2 * crop_samples:
        seg_signal_cropped = seg_signal[:, crop_samples:-crop_samples]
        seg_noise_cropped = seg_noise[:, crop_samples:-crop_samples]
    else:
        # Если блок слишком короткий, не обрезаем (или пропускаем)
        seg_signal_cropped = seg_signal
        seg_noise_cropped = seg_noise
    
    # Добавляем в общий список
    signal_pieces.append(seg_signal_cropped)
    noise_pieces.append(seg_noise_cropped)

# Сшиваем все куски в один длинный массив
if signal_pieces:
    signal_concatenated = np.concatenate(signal_pieces, axis=1)
    noise_concatenated = np.concatenate(noise_pieces, axis=1)
    print(f"Сшито {len(signal_pieces)} блоков, общая длина: {signal_concatenated.shape[1]} отсчётов.")
else:
    raise ValueError("Нет подходящих блоков для SSD!")

# Вычисляем ковариации по сшитому сигналу
C_signal = np.cov(signal_concatenated)
C_noise = np.cov(noise_concatenated)

# Регуляризация шумовой ковариации
reg_coeff = 1e-5
C_noise_reg = C_noise + reg_coeff * np.trace(C_noise) * np.eye(C_noise.shape[0])

# Обобщённая проблема собственных значений
eigvals, eigvecs = eigh(C_signal, C_noise_reg)
idx_sorted = np.argsort(eigvals)[::-1]
n_components_ssd = 30
W_ssd = eigvecs[:, idx_sorted[:n_components_ssd]]
A_ssd = C_signal @ W_ssd

print(f"SSD выполнено на сшитых блоках, получено {n_components_ssd} компонент.")

# %% 
Wsize = 2
Ssize = 0.5
overlap = Wsize - Ssize 

X_windows_band = []
X_windows_ssd = []
X_windows_unfilt = []
 
labels = []
trials = [] 
trial_id = 1

raw_band = raw.copy().filter(l_freq=15, h_freq=25).pick_types(eeg=True)
raw_unfilt = raw.copy().pick_types(eeg=True)

for annot in raw.annotations:
    desc = annot['description']
        
    base_cond = desc

    if annot['duration'] < Wsize or desc == 'BAD_':
    # if annot['duration'] < Wsize or desc == 'BAD_' or desc == 'RS_EC_1' or desc == 'RS_EC_2':
        continue
    print(desc)

    tmin = annot['onset']
    tmax = tmin + annot['duration']
    if tmax > raw.times[-1]:
        tmax = raw.times[-1]
    
    raw_crop = raw_band.copy().crop(tmin=tmin, tmax=tmax)
    raw_crop_unfilt = raw_unfilt.copy().crop(tmin=tmin, tmax=tmax)
    
    epochs_band = mne.make_fixed_length_epochs(
        raw_crop, 
        duration=Wsize, 
        overlap=overlap, 
        preload=True, 
        reject_by_annotation=True, 
        verbose=False
    )
    epochs_band.drop_bad(verbose=False)

    epochs_ssd = mne.make_fixed_length_epochs(
        raw_crop, 
        duration=Wsize, 
        overlap=overlap, 
        preload=True, 
        reject_by_annotation=True, 
        verbose=False
    )
    epochs_ssd.drop_bad(verbose=False)

    epochs_unfilt = mne.make_fixed_length_epochs(
        raw_crop_unfilt, 
        duration=Wsize, 
        overlap=overlap, 
        preload=True, 
        reject_by_annotation=True, 
        verbose=False
    )
    epochs_unfilt.drop_bad(verbose=False)
    
    if epochs_band:            
        if len(epochs_band) > 0 and len(epochs_band) == len(epochs_unfilt):
            X_windows_band.append(epochs_band.get_data(copy=False))
            X_windows_unfilt.append(epochs_unfilt.get_data(copy=False)) 
            labels.extend([base_cond] * len(epochs_band))
            trials.extend([trial_id] * len(epochs_band))
            trial_id += 1 

if len(X_windows_band) > 0:
    X_windows_band = np.concatenate(X_windows_band, axis=0) 
    X_windows_unfilt = np.concatenate(X_windows_unfilt, axis=0)
    labels = np.array(labels)
    trials = np.array(trials)
else:
    raise ValueError("После удаления артефактов не осталось ни одного чистого окна!")

# %%
# =====================================================================
# ПРОЕКЦИЯ ЭПОХ В SSD-ПРОСТРАНСТВО
# =====================================================================
# X_windows_band имеет размер (n_windows, n_channels, n_times)
n_windows, n_ch, n_times = X_windows_band.shape
X_windows_ssd_proj = np.zeros((n_windows, n_components_ssd, n_times))
for i in range(n_windows):
    X_windows_ssd_proj[i] = W_ssd.T @ X_windows_band[i]   # (n_comp, time)

print(f"Эпохи спроецированы в SSD-пространство, размер: {X_windows_ssd_proj.shape}")

# %%
covmats_band = Covariances(estimator='oas').fit_transform(X_windows_band)

print("Вычисление ковариаций на SSD-эпохах...")
covmats_ssd = Covariances(estimator='oas').fit_transform(X_windows_ssd_proj)

print("Проекция в касательное пространство...")
ts_data_ssd = TangentSpace(metric='riemann').fit_transform(covmats_ssd)

Cmean_ssd = mean_riemann(covmats_ssd)

# %%
import time
from pyriemann.geometry.distance import distance, pairwise_distance

# # Замер для Варианта 1 (две матрицы)
# start_time = time.perf_counter()
# dist_0_1 = distance(covmats_ssd[0], covmats_ssd[1], metric='riemann')
# end_time = time.perf_counter()

# print(f"Время выполнения distance: {end_time - start_time:.6f} секунд")


# # Замер для Варианта 2 (все эпохи)
# start_time = time.perf_counter()
# dist_matrix_ssd = pairwise_distance(covmats_ssd, metric='riemann')
# end_time = time.perf_counter()

# print(f"Время выполнения pairwise_distance: {end_time - start_time:.6f} секунд")

# %%
from scipy.linalg import inv, eigh
from umap.umap_ import fuzzy_simplicial_set
from scipy.linalg import null_space 

N_neig = 100
dist_matrix_ssd = pairwise_distance(ts_data_ssd, metric='euclid')

N = 5 # Количество извлекаемых компонентов
current_dim = covmats_ssd.shape[1]          # исходная размерность после SSD
covmats_current = covmats_ssd.copy()
ts_current = TangentSpace(metric='riemann').fit_transform(covmats_current)

filters_full = []        # усредненные фильтры в исходном пространстве 
patterns_full = []       # глобальные паттерны в исходном пространстве
losses = []              
umap_coords_history = [] 
power_history = []       
dims_history = []        

B_cumulative = np.eye(current_dim)
N_epochs = covmats_current.shape[0]

# Исходная матрица расстояний для первого шага
dist_matrix_current = dist_matrix_ssd.copy()

for comp in range(N):
    print(f"\n=== Компонент {comp+1} (текущая размерность {current_dim}) ===")

    # 1. ОБНОВЛЕНИЕ ГРАФА И ЛОКАЛЬНОЕ ОТБЕЛИВАНИЕ
    random_state = np.random.RandomState(42)
    v_ij_sparse, _, _ = fuzzy_simplicial_set(
        X=dist_matrix_current, n_neighbors=N_neig, random_state=random_state,
        metric='precomputed',
    )
    V_graph = v_ij_sparse.toarray()
    np.fill_diagonal(V_graph, 0.0)
    V_graph = V_graph / V_graph.sum(axis=1, keepdims=True)

    B_inv_list = np.zeros_like(covmats_current)
    covmats_pseudo = np.zeros_like(covmats_current) # Это B^{-1} C B^{-1}

    for i in range(N_epochs):
        # Локальный фон B_i = взвешенная сумма соседей
        B_i = np.tensordot(V_graph[i], covmats_current, axes=1)
        
        # Регуляризация фона для безопасного обращения
        # B_i += 1e-6 * np.trace(B_i) * np.eye(current_dim)
        
        b_inv = inv(B_i)
        B_inv_list[i] = b_inv

        # Создаем псевдоковариацию для поиска ПАТТЕРНА
        pseudo_c = b_inv @ covmats_current[i] @ b_inv
        
        # Защита от потери положительной определенности из-за перемножений
        # pseudo_c += 1e-6 * np.trace(pseudo_c) * np.eye(current_dim)
        covmats_pseudo[i] = pseudo_c

    # ИЗМЕНЕНИЕ 2: Используем fit_patterns и передаем C_num и C_den
    # 2. ПОИСК ГЛОБАЛЬНОГО ПАТТЕРНА (a)
    a_opt, final_losses, loss_history = fit_patterns(
        C_num=torch.from_numpy(covmats_pseudo).float(),
        C_den=torch.from_numpy(B_inv_list).float(),
        D_matrix=dist_matrix_current, 
        N_dim=2,
        K_restarts=1,
        n_neighbors=N_neig,
        epochs=100,
        lr=0.05
    )
    losses.append(loss_history)
    a_current = a_opt[0, 0, :] 

    # 3. ВОССТАНОВЛЕНИЕ ДИНАМИЧЕСКИХ ФИЛЬТРОВ И МОЩНОСТИ
    W_dyn_current = np.zeros((N_epochs, current_dim))
    p_comp = np.zeros(N_epochs)
    
    for i in range(N_epochs):
        # Базовый динамический фильтр: w_i = B_i^{-1} a
        w_raw = B_inv_list[i] @ a_current
        
        # КРИТИЧНО: Ограничение единичного усиления (чтобы масштаб не плавал)
        # Фильтр должен пропускать паттерн 'a' с коэффициентом ровно 1.
        w_norm = w_raw / (np.dot(a_current, w_raw))
        W_dyn_current[i] = w_norm
          
        # Оценка мощности в исходном (SSD) пространстве
        w_dyn_full = B_cumulative @ w_norm
        p_comp[i] = np.log(w_dyn_full.T @ covmats_ssd[i] @ w_dyn_full)
        
    power_history.append(p_comp)

    # Сохраняем средний фильтр и текущий паттерн для графиков
    w_avg_current = W_dyn_current.mean(axis=0)
    filters_full.append(B_cumulative @ w_avg_current) 
    patterns_full.append(B_cumulative @ a_current)

    # UMAP для текущего касательного пространства
    if ts_current.shape[1] >= 2:
        umap_step = umap.UMAP(n_components=2)
        coords = umap_step.fit_transform(ts_current)
    else:
        coords = None
    umap_coords_history.append(coords)
    dims_history.append(current_dim)

    # 4. НЕЛИНЕЙНАЯ ДЕФЛЯЦИЯ ПО ЛОКАЛЬНЫМ ПАТТЕРНАМ
    if comp < N - 1:
        # Находим базис (матрицу Pu), ортогональный глобальному паттерну a_current.
        # a_current имеет размер (current_dim,), null_space вернет матрицу (current_dim, current_dim-1)
        Pu = null_space(a_current.reshape(1, -1))
        
        covmats_new = np.zeros((N_epochs, current_dim-1, current_dim-1))
        for i in range(N_epochs):
            # Жестко отсекаем направление 'a' из каждой ковариации
            c_new = Pu.T @ covmats_current[i] @ Pu
            
            # Легкая регуляризация для защиты численной стабильности (SPD)
            covmats_new[i] = c_new + 1e-6 * np.trace(c_new) * np.eye(current_dim - 1)
            
        # Обновляем кумулятивную матрицу проекций
        B_cumulative = B_cumulative @ Pu
        current_dim -= 1
        covmats_current = covmats_new
        
        # Пересчет геометрии (теперь безопасно!)
        ts_current = TangentSpace(metric='riemann').fit_transform(covmats_current)
        dist_matrix_current = pairwise_distance(ts_current, metric='euclid')

    print(f"  Потери: {final_losses[-1]:.4f}")

# %%
# ========== График 1: кривые обучения ==========
plt.figure(figsize=(10, 6))
for idx, loss in enumerate(losses):
    plt.plot(loss, label=f'Компонент {idx+1}')
plt.xlabel('Эпоха оптимизации')
plt.ylabel('Loss')
plt.title('Эволюция функции потерь для каждого фильтра')
plt.legend()
plt.grid(True)
plt.show()

# ========== График 2: UMAP на каждом шаге дефляции ==========
n_steps = len(umap_coords_history)
cols = int(np.ceil(n_steps / 2))
fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
axes = axes.flatten()
for step, (coords, dim) in enumerate(zip(umap_coords_history, dims_history)):
    ax = axes[step]
    if coords is None:
        ax.text(0.5, 0.5, f'Шаг {step}\n(одномерное пространство)', ha='center', va='center')
        ax.set_title(f'Шаг {step}')
    else:
        p_vals = power_history[step]
        p_z = (p_vals - p_vals.mean()) / p_vals.std()
        ax.scatter(coords[:, 0], coords[:, 1], c=p_z, cmap='plasma', s=10)
        ax.set_title(f'Шаг {step} (размерность {dim})')
    ax.set_xticks([])
    ax.set_yticks([])
# Скрываем лишние панели, если есть
for ax in axes[n_steps:]:
    ax.set_visible(False)
plt.suptitle('Эволюция топологии касательного пространства по шагам дефляции', fontsize=14)
plt.tight_layout()
plt.show()

# %%
reducer = umap.UMAP(n_components=2, metric='precomputed')
umap_coords = reducer.fit_transform(dist_matrix_ssd)

# %%
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

comp_idx = 4 # выберите нужный компонент

w_comp = filters_full[comp_idx] 
a_comp = patterns_full[comp_idx] # Достаем найденный паттерн

# Проецируем в пространство сенсоров через матрицу SSD
A_pattern = W_ssd @ a_comp     # Паттерн (Forward Model)
W_sensor = W_ssd @ w_comp      # Фильтр (Inverse Model)

p_vals = power_history[comp_idx]
p_z = (p_vals - p_vals.mean()) / p_vals.std()

# UMAP дефлированного пространства для данного этапа
umap_defl = umap_coords_history[comp_idx]   

# Порядок условий: сохраняем порядок появления, не сортируем
unique_labels_ordered = []
for lab in labels:
    if lab not in unique_labels_ordered:
        unique_labels_ordered.append(lab)

# Цветовая палитра
cmap = plt.cm.get_cmap('tab10', len(unique_labels_ordered))
label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels_ordered)}

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])

# ---- Верхний ряд ----
# 1. UMAP исходного недефлированного пространства
ax_umap_orig = fig.add_subplot(gs[0, 0])
sc1 = ax_umap_orig.scatter(umap_coords[:, 0], umap_coords[:, 1],
                           c=p_z, cmap='plasma', s=15)
plt.colorbar(sc1, ax=ax_umap_orig, label='z-score log-power')
ax_umap_orig.set_title(f'Компонент {comp_idx+1}\nUMAP исходного пр-ва')
ax_umap_orig.set_xticks([])
ax_umap_orig.set_yticks([])

# 2. Топопаттерн
ax_topo = fig.add_subplot(gs[0, 1])
mne.viz.plot_topomap(A_pattern, raw.info, axes=ax_topo, show=False)
ax_topo.set_title('Топографический паттерн (Forward Model)')

# 3. UMAP дефлированного пространства (если доступен)
ax_umap_defl = fig.add_subplot(gs[0, 2])
if umap_defl is not None:
    sc2 = ax_umap_defl.scatter(umap_defl[:, 0], umap_defl[:, 1],
                               c=p_z, cmap='plasma', s=15)
    plt.colorbar(sc2, ax=ax_umap_defl, label='z-score log-power')
    ax_umap_defl.set_title(f'UMAP дефлированного пр-ва\n(шаг {comp_idx})')
else:
    ax_umap_defl.text(0.5, 0.5, 'Размерность < 2\n(нет вложения)',
                      ha='center', va='center', transform=ax_umap_defl.transAxes)
    ax_umap_defl.set_title(f'Дефлированное пр-во (шаг {comp_idx})')
ax_umap_defl.set_xticks([])
ax_umap_defl.set_yticks([])

# ---- Нижний ряд ----
# 4. Пространственный фильтр (сенсорное пространство)
ax_filt = fig.add_subplot(gs[1, 0])
mne.viz.plot_topomap(W_sensor, raw.info, axes=ax_filt, show=False)
ax_filt.set_title('Усредненный пространственный фильтр')

# 5. Огибающая мощности с подписями условий на оси Ox
ax_env = fig.add_subplot(gs[1, 1])
window_idx = np.arange(len(p_z))
ax_env.plot(window_idx, p_z, color='black', lw=0.8)

# Цветная заливка по условиям и сбор меток для оси X
xtick_positions = []
xtick_labels = []
for lab in unique_labels_ordered:
    mask = (labels == lab)
    if not np.any(mask):
        continue
    changes = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        ax_env.axvspan(s, e-1, facecolor=label_to_color[lab], alpha=0.2, zorder=0)
        mid = (s + e - 1) / 2
        xtick_positions.append(mid)
        xtick_labels.append(lab)

ax_env.set_xticks(xtick_positions)
ax_env.set_xticklabels(xtick_labels, rotation=90, ha='center', fontsize=8)
ax_env.set_xlabel('Номер окна')
ax_env.set_ylabel('z-score log-power')
ax_env.set_title('Огибающая мощности источника')

# 6. Правая нижняя панель
ax_extra = fig.add_subplot(gs[1, 2])
ax_extra.axis('off')

plt.suptitle(f'Компонент {comp_idx+1} — Анализ топологии и топографии', fontsize=16)
plt.tight_layout()
plt.show()

# %%s
# Предполагаем, что у вас уже есть:
# raw_ica — исходный Raw после ICA (или любой другой)
# W_ssd — матрица SSD (n_channels, n_components_ssd)
# filters_full — массив (n_filters, n_components_ssd)

# 1. Берём только EEG-каналы (чтобы размерность совпадала с W_ssd)
raw_eeg = raw.copy().pick_types(eeg=True)
data = raw_eeg.get_data()          # (n_channels, n_samples)
sfreq = raw_eeg.info['sfreq']

# 2. Матрица проекции из сенсорного пространства в компоненты
P = W_ssd @ np.array(filters_full).T         # (n_channels, n_filters)
components = P.T @ data            # (n_filters, n_samples)

# 3. Создаём info для новых каналов
n_filters = components.shape[0]
ch_names = [f'Comp_{i+1:02d}' for i in range(n_filters)]
ch_types = ['eeg'] * n_filters 
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# 4. Создаём RawArray
raw_components = mne.io.RawArray(components, info)

# 5. Переносим аннотации (если нужно)
new_ann = mne.Annotations(raw_eeg.annotations.onset,raw_eeg.annotations.duration,raw_eeg.annotations.description)
raw_components.set_annotations(new_ann)

# 6. Визуализация
raw_components.copy().filter(l_freq=15, h_freq=25).plot(
    picks='all',  # <--- Ключевое исправление
    n_channels=n_filters, 
    scalings='auto', 
    title='Выделенные компоненты'
)
# Или по отдельности:
# raw_components.plot_psd()
