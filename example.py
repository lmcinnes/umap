# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:07:11 2025

@author: anton
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
import umap
import scipy.io
from pyriemann.utils.mean import mean_riemann
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh

# %%
import sys
sys.path.append("C:/Users/ansbel/Documents/GitHub/site-packages/umap_meeg")
from topological_spatial_filter import fit_filters
import torch

# %%
fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part1/eeg/10_07_g1_2223_raw.fif"
# fpath = "C:/Users/ansbel/Documents/GitHub/TriCo/data/external/music_listening/part2/eeg/TumAle_raw.fif"

raw = mne.io.read_raw_fif(fpath,preload=True)
sfreq = raw.info['sfreq']

# %%
# raw_filt = raw.copy().notch_filter(50).filter(l_freq=0.1,h_freq=70)
raw_filt = raw

# %% 
raw_filt.plot(n_channels=38)

# %%
raw_interpolated = raw_filt.copy().interpolate_bads(reset_bads=True)

# %%
raw_interpolated.plot(n_channels=38)

# %%
raw_interpolated.save()

# %%
ica = ICA(
    n_components=0.999,     
    method='fastica',      
    random_state=42,
    max_iter='auto'
)

ica.fit(raw_interpolated)

# %%
ica.plot_components()
ica.plot_sources(raw_filt)

# %%
raw_ica = ica.apply(raw_interpolated)

# %%
raw_ica.plot(n_channels=38)

# %%
raw_ica = raw

# %%
new_durations = [
    ann['duration'] if ann['description'].startswith('BAD_') else 120.0
    for ann in raw.annotations
]

# Создаем новые аннотации и применяем их к объекту
new_ann = mne.Annotations(
    onset=raw.annotations.onset,
    duration=new_durations,
    description=raw.annotations.description,
    orig_time=raw.annotations.orig_time  # важно сохранить исходную временную привязку
)

raw_ica.set_annotations(new_ann)

print("Длительности аннотаций успешно обновлены!")

# %%
# new_descriptions = [
#     'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
#     'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
#     'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
#     'Waltz_6', 'Waltz_7', 'Waltz_8'
# ]
new_descriptions = [
    'RS_EC_1', 'RS_EO_1', '2Hz', '05Hz', '4Hz', '1Hz', '3Hz',
    'NoRy_1', 'Waltz_1', 'Waltz_2', 'NoRy_2', 'NoRy_3', 'Waltz_3',
    'NoRy_4', 'Waltz_4', 'NoRy_5', 'Waltz_5', 'RS_EC_2', 'RS_EO_2',
]

descriptions = raw_ica.annotations.description

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
old_annot = raw_ica.annotations
new_annot = mne.Annotations(
    onset=old_annot.onset,
    duration=old_annot.duration,
    description=np.array(new_desc, dtype='U20'),  # тип с запасом по длине
    orig_time=old_annot.orig_time
)

# Применяем новые аннотации к raw_ica
raw_ica.set_annotations(new_annot)

# Проверяем изменение
print(raw_ica.annotations.description)

# %% 
# Получаем данные после ICA (все каналы EEG)
raw_clean = raw_ica.copy().pick_types(eeg=True)
data = raw_ica.get_data()  

# Списки для хранения отфильтрованных и обрезанных кусков
signal_pieces = []
noise_pieces = []
times_pieces = []

# Параметры фильтров
b_signal, a_signal = butter(3, np.array([15, 25]) / (int(sfreq) / 2), btype='band')
b_broad, a_broad = butter(3, np.array([13, 30]) / (int(sfreq) / 2), btype='band')
b_stop, a_stop = butter(3, np.array([14.5, 25.5]) / (int(sfreq) / 2), btype='stop')
# b_signal, a_signal = butter(3, np.array([8, 12]) / (int(sfreq) / 2), btype='band')
# b_broad, a_broad = butter(3, np.array([6, 14]) / (int(sfreq) / 2), btype='band')
# b_stop, a_stop = butter(3, np.array([7.5, 12.5]) / (int(sfreq) / 2), btype='stop')

# Длительность для обрезания краёв (в секундах)
crop_duration = 0.5  # 0.5 секунды с каждой стороны
crop_samples = int(crop_duration * sfreq)

# Проходим по аннотациям
for annot in raw_ica.annotations:
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
    if tmax > raw_ica.times[-1]:
        tmax = raw_ica.times[-1]
    
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

raw_band = raw_ica.copy().filter(l_freq=15, h_freq=25).pick_types(eeg=True)
raw_unfilt = raw_ica.copy().pick_types(eeg=True)

for annot in raw_ica.annotations:
    desc = annot['description']
        
    base_cond = desc

    if annot['duration'] < Wsize or desc == 'BAD_':
    # if annot['duration'] < Wsize or desc == 'BAD_' or desc == 'RS_EC_1' or desc == 'RS_EC_2':
        continue
    print(desc)

    tmin = annot['onset']
    tmax = tmin + annot['duration']
    if tmax > raw_ica.times[-1]:
        tmax = raw_ica.times[-1]
    
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
N = 5
current_dim = covmats_ssd.shape[1]          # исходная размерность после SSD
covmats_current = covmats_ssd.copy()
ts_current = TangentSpace(metric='riemann').fit_transform(covmats_current)

filters_full = []   # итоговые фильтры в полном пространстве
losses = []

# Матрица, которая преобразует из полного пространства в текущее редуцированное.
# В начале это просто единичная матрица.
B_cumulative = np.eye(current_dim)

for comp in range(N):
    print(f"\n=== Компонент {comp+1} (текущая размерность {current_dim}) ===")
    
    # 1. Ищем один фильтр в текущем пространстве
    w_opt, final_losses, loss_history = fit_filters(
        C=torch.from_numpy(covmats_current).float(),
        T_features=torch.from_numpy(ts_current).float(),
        K=1,
        n_neighbors=200,
        epochs=50,
        lr=0.1
    )
    losses.append(final_losses)
    v_np = w_opt[0, :, 0].detach().cpu().numpy()   
    
    w_full = B_cumulative @ v_np
    filters_full.append(w_full)
    
    if comp < N - 1:
        v_unit = v_np / np.linalg.norm(v_np)
        Q, _ = np.linalg.qr(np.column_stack([v_unit, np.eye(current_dim)]))
        B_local = Q[:, 1:]   # shape (current_dim, current_dim-1)
        
        covmats_new = np.zeros((covmats_current.shape[0], current_dim-1, current_dim-1))
        for i in range(covmats_current.shape[0]):
            covmats_new[i] = B_local.T @ covmats_current[i] @ B_local
        
        B_cumulative = B_cumulative @ B_local
        current_dim = current_dim - 1
        covmats_current = covmats_new
        
        ts_current = TangentSpace(metric='riemann').fit_transform(covmats_current)
    
    print(f"  Потери: {final_losses[-1]:.4f}")

# %%
reducer = umap.UMAP(n_components=2, random_state=42)
umap_coords = reducer.fit_transform(ts_data_ssd)

# %% =====================================================================
# =====================================================================
# ВИЗУАЛИЗАЦИЯ
# =====================================================================
filters_full = np.array(filters_full)

w_vector = filters_full[4, :]
best_w_np = w_vector
p_source = np.zeros(covmats_ssd.shape[0])
for i in range(covmats_ssd.shape[0]):
    p_source[i] = np.log(best_w_np.T @ covmats_ssd[i] @ best_w_np)

p_source_z = (p_source - p_source.mean()) / p_source.std()

plt.figure()
plt.plot(p_source)
plt.title('Мощность источника по окнам')
plt.show()

plt.figure()
plt.scatter(umap_coords[:, 0], umap_coords[:, 1],
                   c=p_source_z, cmap='plasma', s=20)

A = A_ssd @ best_w_np

fig_topo, ax_topo = plt.subplots()
mne.viz.plot_topomap(A, raw_ica.info, axes=ax_topo, show=True)
ax_topo.set_title('Паттерн выделенного источника (сенсорное пространство)')
plt.show()


# %%
# Предполагаем, что у вас уже есть:
# raw_ica — исходный Raw после ICA (или любой другой)
# W_ssd — матрица SSD (n_channels, n_components_ssd)
# filters_full — массив (n_filters, n_components_ssd)

# 1. Берём только EEG-каналы (чтобы размерность совпадала с W_ssd)
raw_eeg = raw_ica.copy().pick_types(eeg=True)
data = raw_eeg.get_data()          # (n_channels, n_samples)
sfreq = raw_eeg.info['sfreq']

# 2. Матрица проекции из сенсорного пространства в компоненты
P = W_ssd @ filters_full.T         # (n_channels, n_filters)
components = P.T @ data            # (n_filters, n_samples)

# 3. Создаём info для новых каналов
n_filters = components.shape[0]
ch_names = [f'Comp_{i+1:02d}' for i in range(n_filters)]
ch_types = ['misc'] * n_filters   # можно также 'eeg', но тогда MNE будет ожидать стандартные позиции
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# 4. Создаём RawArray
raw_components = mne.io.RawArray(components, info)

# 5. Переносим аннотации (если нужно)
new_ann = mne.Annotations(raw_eeg.annotations.onset,raw_eeg.annotations.duration,raw_eeg.annotations.description)
raw_components.set_annotations(new_ann)

# 6. Визуализация
raw_components.plot(n_channels=n_filters, scalings='auto', title='Выделенные компоненты')
# Или по отдельности:
# raw_components.plot_psd()
