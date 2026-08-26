import numpy as np
import random
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics import mean_squared_error
import pywt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import torch
import gc

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
import numpy as np
import time
#import up_data
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

import numpy.linalg as LA
from scipy.stats import _entropy
from sys import exit

from scipy.signal import butter, sosfilt,wiener

from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.decomposition import PCA


#KS
def ks(x, y, test_size=0.2):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size (float)
    :return: spec_train: (n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    M = x.shape[0]
    N = round((1-test_size) * M)
    samples = np.arange(M)

    D = np.zeros((M, M))

    for i in range((M-1)):
        xa = x[i, :]
        for j in range((i+1), M):
            xb = x[j, :]
            D[i, j] = np.linalg.norm(xa-xb)

    maxD = np.max(D, axis=0)
    index_row = np.argmax(D, axis=0)
    index_column = np.argmax(maxD)

    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros((M-i))
        for j in range((M-i)):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y[m]
    spec_test = x[m_complement, :]
    target_test = y[m_complement]
    return spec_train, spec_test, target_train, target_test

def fast_ks(features, labels, test_size):
    """
    Optimized fast Kennard-Stone algorithm supporting multi-label data

    Parameters:
    features : pandas.DataFrame, shape (n_samples, n_features)
        Input feature DataFrame
    labels : pandas.DataFrame, shape (n_samples, n_labels)
        Target label DataFrame (supports multi-label)
    test_size : float or int
        Test set proportion (0.0–1.0) or number of samples

    Returns:
    tuple: (X_train, X_test, y_train, y_test), all of which are DataFrames
    """


    n = features.shape[0]
        
    if labels.ndim == 1:
            # labels = labels.to_frame()  # DataFrame
            labels = labels.reshape(-1, 1) 
        
    if isinstance(test_size, float):
            n_test = max(1, int(n * test_size))
    else:
            n_test = min(test_size, n - 1)
            
    if n_test <= 0 or n_test >= n:
            raise ValueError(f"Invalid test set size: {n_test}. It must be between 1 and {n-1}")
        
        # dist_matrix = pairwise_distances(features.values, metric='euclidean')
    dist_matrix = pairwise_distances(features, metric='euclidean')
        
        # idx1, idx2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    idx1, idx2 = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    test_indices = [idx1, idx2]
        
    min_dists = np.min(dist_matrix[:, test_indices], axis=1)
    np.fill_diagonal(dist_matrix, np.inf)  
        
    for _ in range(3, n_test + 1):
            # Find the sample with the largest minimum distance
            candidate = np.argmax(min_dists)
            test_indices.append(candidate)
            
            new_dists = dist_matrix[:, candidate]
            min_dists = np.minimum(min_dists, new_dists)
            min_dists[candidate] = -np.inf  
        
    train_indices = list(set(range(n)) - set(test_indices))
        
    return (
            # features.iloc[train_indices], features.iloc[test_indices],
            # labels.iloc[train_indices], labels.iloc[test_indices]
            features[train_indices], features[test_indices],
            labels[train_indices], labels[test_indices]
        )


# SPA
class SPA:
    def spa(self, spectrum, spectrum_nums, spectrum_range, random_seed, if_random, intial_index):
        """spa

        Args:
            spectrum (_type_): Raw spectrum
            spectrum_nums (_type_): Number of bins
            spectrum_range (_type_): Band range
            random_seed (_type_): Random seed
            if_random (_type_): Whether to use random values
            initial_index (_type_): Initialization list

        Returns:
            _type_: _description_
        """
   
        N = spectrum_nums
        temp_dict = {}
        for num in range(spectrum.shape[1]):
            temp_dict[num] = spectrum[:, num]
        # print(temp_dict)

        if if_random is True:
            random.seed(random_seed)
            inital_index1 = random.randrange(spectrum_range[0], spectrum_range[1] + 1, 1)
        else:
            inital_index1 = intial_index
  
        inital_spectrum = temp_dict[inital_index1]

        del temp_dict[inital_index1]
        index_list = []

        index_list.append(inital_index1)

        n = 1
        while (n < N):
            temp_p_norm = 'start'
            max_norm_index = -1
            for key, value in temp_dict.items():

                p = np.linalg.norm(inital_spectrum - np.dot(np.dot((np.dot(inital_spectrum.T, value)), value),
                                                            pow(np.dot(value.T, value), -1)))

                if temp_p_norm == 'start':
                    temp_p_norm = p
                    max_norm_index = key

                else:
                    if p > temp_p_norm:
                        temp_p_norm = p
                        max_norm_index = key
                    else:
                        pass
            index_list.append(max_norm_index)
            inital_spectrum = temp_dict[max_norm_index]
            del temp_dict[max_norm_index]
            n = n + 1
        return sorted(index_list)


# MSC
def do_msc(input_data):

    ref_spectrum = np.mean(input_data, axis=0)
    corrected_data = np.zeros_like(input_data)
 
    for i in range(input_data.shape[0]):

        fit = np.polyfit(ref_spectrum, input_data[i, :], 1)

        corrected_data[i, :] = (input_data[i, :] - fit[1]) / fit[0]
 
    return corrected_data, ref_spectrum

# Deconvolution function
def deconvolve_spectra(spectra):

    return wiener(spectra, mysize=None, noise=None)


# RPD
def calculate_rpd(y_test, pred):
  
    sd = np.std(y_test)

    rmse = mean_squared_error(y_test, pred, squared=False)

    # RPD
    rpd = sd / rmse

    return rpd

# RPIQ
def calculate_rpiq(predicted, observed):
    # predicted
    # observed
    iqr = np.subtract(*np.percentile(observed, [75, 25]))
    #print(observed)  
    #print(np.unique(observed))  
    #print(np.percentile(observed, [25, 75])) 
    #print(iqr)

    rmsep = np.sqrt(mean_squared_error(observed, predicted))

    rpiq = iqr / rmsep

    return rpiq

'''CARS'''
def cars(X, y, num_selected=15, n_folds=5, alpha=0.8, random_state=None):
        """
        A Robust CARS Implementation
        Parameters:
            X: Spectral data (n_samples, n_bands)
            y: Target variable (n_samples,)
            num_selected: Number of bands to select (must be < n_bands)
            random_state: Random seed
        Returns:
            Indices of valid bands (sorted)
        """
        np.random.seed(random_state)
        n_samples, n_bands = X.shape
        assert num_selected < n_bands, "num_selected必须小于波段总数"
        
        with np.errstate(all='ignore'):
            abs_corr = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_bands)])
        weights = np.nan_to_num(abs_corr, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 1e-8, 1)  
        
        selected_bands = []
        
        for _ in range(num_selected):

            prob = weights / (weights.sum() + 1e-8)
            try:
                candidates = np.random.choice(
                    n_bands, 
                    size=min(50, n_bands), 
                    p=prob, 
                    replace=False
                )
            except:
                candidates = np.argsort(weights)[-50:]  
            
            best_band = None
            best_score = -np.inf
            
            for band in candidates:
                if band >= n_bands or band in selected_bands:
                    continue
                    
                temp_bands = selected_bands + [band]
                X_sub = X[:, temp_bands]
                

                pls = PLSRegression(n_components=min(5, len(temp_bands)))
                score = np.mean(cross_val_score(pls, X_sub, y, cv=n_folds, scoring='r2'))
                
                if score > best_score:
                    best_score = score
                    best_band = band
            
            if best_band is not None:
                selected_bands.append(best_band)
                weights[best_band] = 0 
                weights *= np.exp(-alpha * weights / (weights.max() + 1e-8))
        
        valid_bands = sorted([b for b in selected_bands if b < n_bands])
        return valid_bands[:num_selected]

''' Continuous Wavelet Transform '''
def CWT(data):
    # wavelet = pywt.Wavelet('bior1.3')
    wavelet = pywt.ContinuousWavelet('gaus6')
    # # scales = np.concatenate((np.arange(1, 11), 2 ** np.arange(1, 11)))
    # # scales = np.arange(1, 25,4)
    scales = np.arange(1, 20)
    cwtmatr, freqs = pywt.cwt(data[0], scales, wavelet)

    return cwtmatr,freqs

def select_frequencies(cwtmatr, freqs, num_bands=40):

    max_indices = np.unravel_index(np.argsort(np.abs(cwtmatr), axis=None)[-1::-1], cwtmatr.shape)

    selected_indices = max_indices[1][-num_bands:]

    return selected_indices


''' Discrete Wavelet Transform '''
def DWT(spectra, wavelet_name='db4', max_level=3):

    coeffs_list = [pywt.wavedec(spectra[i, :], wavelet_name, level=max_level) for i in range(spectra.shape[0])]
    
    threshold = 0.1
    denoised_coeffs_list = []
    for coeffs in coeffs_list:

        cA, *cD = coeffs
   
        thresholded_cD = [pywt.threshold(i, value=threshold, mode='soft') for i in cD]
     
        denoised_coeffs = [cA] + thresholded_cD
        denoised_coeffs_list.append(denoised_coeffs)
    
    denoised_spectra = np.array([pywt.waverec(denoised_coeffs, wavelet_name) for denoised_coeffs in denoised_coeffs_list])
    
    return denoised_spectra

'''IRIV'''
class IRIV:
    '''Iteratively retaining informative variables (IRIV)'''

    def __init__(self, max_components=10):
        self.max_components = max_components
        self.back_elimination = Elimination(max_components)

    def iteration(self, data, label, iter_num=100, min_dimension=30):
        self.data, self.label = data, label
        for j in range(iter_num):
            start_time = time.time()
            store_variables, remove_variables = self.__calculate_informative_variable(data, label)
            if np.sum(remove_variables) == 0 or np.sum(store_variables) <= min_dimension:
                data = data[:, store_variables]   
                print(
                    'The iterative rounds of IRIV have been finished, now enter into the process of backward elimination!\n')
                break

            data = data[:, store_variables]
            print('The %d th round of IRIV has finished!  ' % (j + 1))
            print('Remain %d / %d  variable, using time: %g seconds!\n' % (data.shape[1],
                                                                           self.data.shape[1],
                                                                           time.time() - start_time))

        data, _ = self.back_elimination.iteration(data, label) 
        self.remain_data = data
        return data

    def __calculate_informative_variable(self, data, label):
        rmsecv5, A = self.__calculate_rmsecv(data, label)
    
        rmsecv_origin = np.tile(rmsecv5[:, 0].reshape(-1, 1), (A.shape[1],))
      
        rmsecv_replace = rmsecv5[:, 1:]

        rmsecv_exclude = rmsecv_replace.copy()
        rmsecv_include = rmsecv_replace.copy()
        
        rmsecv_exclude[A == 0] = rmsecv_origin[A == 0]
        rmsecv_include[A == 1] = rmsecv_origin[A == 1]
        exclude_mean = np.mean(rmsecv_exclude, axis=0) 
        include_mean = np.mean(rmsecv_include, axis=0)

        p_val, DMEAN, H = [], [], []
        for i in range(A.shape[1]):  
            _, pVal = stats.mannwhitneyu(rmsecv_exclude[:, i], rmsecv_include[:, i],
                                         alternative='two-sided') 
            H.append(int(pVal <= 0.05))  

           
            temp_DMEAN = exclude_mean[i] - include_mean[i]
            if temp_DMEAN < 0:
                pVal = pVal + 1

           
            p_val.append(pVal)
            DMEAN.append(temp_DMEAN)
        p_val = np.stack(p_val)
        DMEAN = np.stack(DMEAN)
        H = np.stack(H)

      
        strong_inform = (H == 1) * (p_val < 1)
        weak_inform = (H == 0) * (p_val < 1)
        un_inform = (H == 0) * (p_val >= 1)
        interfering = (H == 1) * (p_val >= 1)
       
        remove_variables = un_inform | interfering
        store_variables = strong_inform | weak_inform
        return store_variables, remove_variables

    def __calculate_rmsecv(self, data, label):
        
        A, row = self.__generate_binary_matrix(data)

        
        rmsecv5 = np.zeros((row, data.shape[1] + 1))  

      
        for k, sub_a in enumerate(A):
            sub_data = data[:, sub_a == 1]

            n_component = min(np.sum(sub_a == 1), self.max_components)
            model = PLSRegression(n_components=n_component, max_iter=100)
            score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
            rmsecv5[k, 0] = np.sqrt(-score)

        for i in range(data.shape[1]):
            B = np.copy(A)
            B[:, i] = 1 - B[:, i]
            for k, sub_b in enumerate(B):
                sub_data = data[:, sub_b == 1]

                n_component = min(np.sum(sub_b == 1), self.max_components)
                model = PLSRegression(n_components=n_component, max_iter=100)
                score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1,
                                        scoring='neg_mean_squared_error').mean()
                rmsecv5[k, i + 1] = np.sqrt(-score)

            gc.collect()  
        return rmsecv5, A

    def __generate_binary_matrix(self, data):
        
        if data.shape[1] >= 500:  
            row = 500
        elif data.shape[1] >= 300:
            row = 300
        elif data.shape[1] >= 100:
            row = 200
        elif data.shape[1] >= 50:
            row = 100
        else:
            row = 50

        A = np.ones((row, data.shape[1]))  
        A[row // 2:] = 0   

        while True:
            A = np.stack([np.random.permutation(sub_a) for sub_a in A.transpose()]).transpose()
           
            if not np.sum(np.sum(A, axis=1) == 0):
                break
        return A, row

    def remain_index(self):
       
        index = []
        for i in range(self.remain_data.shape[1]):  

            index.append(np.where(np.sum(self.data - self.remain_data[:, i].reshape(-1, 1), axis=0) == 0)[0][0])
        return index


class Elimination:
    def __init__(self, max_components=10):
        self.max_components = max_components

    def iteration(self, data, label):
        delete_index = []
        base_score = self.__get_score(data, label)
        while True:
            scores = self.__get_partial_score(data, label, delete_index)  
            if base_score < scores.min():
                break
            else:
              
                base_score = scores.min()
                index = np.argmin(scores)
                delete_index.append(index)
        return np.delete(data, delete_index, axis=1), delete_index


    def __get_partial_score(self, data, label, delete_index):
       
        scores = []
        for i in range(data.shape[1]):
            if i in delete_index:   
                scores.append(np.inf)
                continue
            index = delete_index.copy()
            index.append(i)
            sub_data = np.delete(data, index, axis=1)

            n_component = min(sub_data.shape[1], self.max_components)
            model = PLSRegression(n_components=n_component, max_iter=1000)
            
            sub_score = cross_val_score(model, sub_data, label, cv=5, n_jobs=-1,
                                        scoring='neg_mean_squared_error').mean()
           
            scores.append(np.sqrt(-sub_score))
        return np.stack(scores)

    def __get_score(self, data, label):
        n_component = min(data.shape[1], self.max_components)
        model = PLSRegression(n_components=n_component, max_iter=1000)
        score = cross_val_score(model, data, label, cv=5, n_jobs=-1, scoring='neg_mean_squared_error').mean()
        return np.sqrt(-score)


def calculate_retained_index(train_data, train_label):
    start_time = time.time()
    iriv_model = IRIV()
    iriv_model.iteration(train_data, train_label)
    retained_index = iriv_model.remain_index()
    end_time = time.time()
    print('Finished calculating the strong and weak information variables, took some time %.3fs' % (end_time - start_time))
    print('Keep index [starting from 0]:', retained_index)

    return retained_index

'''def Getdata():
    x_train, y_train, x_test, y_test = up_data.get_data()
    x_train=np.array(x_train)
    y_train=np.array(y_train)
   
    length=len(fea)
    # print(length)

    x_train = x_train[:, fea]
    x_test = x_test[:, fea]

    return x_train, y_train, x_test, y_test,length'''


def direct_standardization_improved(lab_spectra, sat_spectra):
    """
    Direct Standardization, DS
  
    
    Parameters:
    - lab_spectra: Laboratory spectral data, in the form (n_samples, n_wavelengths)
    - sat_spectra: Satellite spectral data, with shape (n_samples, n_wavelengths)

    Returns:
    - corrected_sat_spectra: Corrected satellite spectral data, with the same shape as sat_spectra
    """


# ==================== DS ====================
    #  X_lab ≈ X_sentinel @ B + E
    pls = PLSRegression(n_components=min(sat_spectra.shape[1], lab_spectra.shape[1]))
    pls.fit(sat_spectra, lab_spectra)
    B = pls.coef_.T  # (n_sentinel, n_lab)
        
    X_pred = pls.predict(sat_spectra)
    E = lab_spectra - X_pred
        
    corrected_sat_spectra = sat_spectra @ B + E.mean(axis=0)
    
    return corrected_sat_spectra

def piecewise_direct_standardization(lab_spectra, sat_spectra, n_segments=5):
    """
    Piecewise Direct Standardization, PDS
    
    Parameters:
    - lab_spectra: Laboratory spectral data, in the form (n_samples, n_wavelengths)
    - sat_spectra: Satellite spectral data, in the form (n_samples, n_wavelengths)
    - n_segments: The number of segments into which the spectra are divided

    Returns:
    - corrected_sat_spectra: Corrected satellite spectral data, with the same shape as sat_spectra
    """
    n_samples, n_wavelengths = lab_spectra.shape
    
    segment_size = n_wavelengths // n_segments
    remainder = n_wavelengths % n_segments
    segments = []
    start = 0
    for i in range(n_segments):
        end = start + segment_size + (1 if i < remainder else 0)
        segments.append((start, end))
        start = end

    corrected_sat_spectra = np.zeros_like(sat_spectra)
    
    # DS
    for seg_start, seg_end in segments:

        lab_seg = lab_spectra[:, seg_start:seg_end]
        sat_seg = sat_spectra[:, seg_start:seg_end]
        
        corrected_seg = direct_standardization_improved(lab_seg, sat_seg)
        
        corrected_sat_spectra[:, seg_start:seg_end] = corrected_seg
    
    return corrected_sat_spectra

def calibrate_pds(X_lab, X_sat, X_sat_to_correct, window_size=30):
    n_features = X_lab.shape[1]
    X_corr = np.zeros_like(X_sat_to_correct)
    for start in range(0, n_features, window_size):
        end = min(start + window_size, n_features)
        model = PLSRegression()
        model.fit(X_sat[:, start:end], X_lab[:, start:end])
        X_corr[:, start:end] = model.predict(X_sat_to_correct[:, start:end])
    return X_corr

def calibrate_epo(X_lab, X_sat, X_sat_to_correct, n_components=6):
    D = X_sat - X_lab  
    pca = PCA(n_components=n_components)
    pca.fit(D)
    P = pca.components_.T  
    return X_sat_to_correct - X_sat_to_correct @ P @ P.T

import numpy as np
from sklearn.cross_decomposition import PLSRegression

def PiecewiseDirectStandardization(master_spectra, slave_spectra, window_half_size, n_components, wavelengths=None, verbose=True):
    """
    Piecewise Direct Standardization (PDS) that directly returns corrected slave spectra.
    
    Parameters:
    - master_spectra: np.ndarray (samples x wavelengths), from lab instrument
    - slave_spectra: np.ndarray (samples x wavelengths), from satellite instrument
    - window_half_size: int, half window size (k+1 = full_window//2)
    - n_components: int, number of latent variables for PLS
    - wavelengths: optional, just for reference
    - verbose: bool, print progress
    
    Returns:
    - corrected_slave: np.ndarray (samples x wavelengths), corrected slave spectra
    """

    if master_spectra.shape != slave_spectra.shape:
        raise ValueError("Master and slave spectra must have the same shape.")

    n_samples, n_wavelengths = master_spectra.shape
    k = window_half_size - 1

    transfer_matrix = np.zeros((n_wavelengths, n_wavelengths - 2 * window_half_size + 2))
    intercept_vector = []

    current_idx = window_half_size
    total_to_process = n_wavelengths - 2 * window_half_size + 1
    processed_count = 0

    while current_idx <= (n_wavelengths - k):
        target = master_spectra[:, current_idx - 1]

        start_idx = max(0, current_idx - 1 - k)
        end_idx = min(n_wavelengths, current_idx + k)
        actual_window_size = end_idx - start_idx

        X = slave_spectra[:, start_idx:end_idx]

        pad_left = pad_right = 0
        if actual_window_size < (2 * k + 1):
            pad_left = max(0, k - (current_idx - 1))
            pad_right = max(0, k - (n_wavelengths - current_idx))
            X = np.pad(X, ((0, 0), (pad_left, pad_right)), mode='constant')

        actual_ncomp = min(n_components, X.shape[1])
        if actual_ncomp < 1:
            intercept_vector.append(0.0)
            current_idx += 1
            continue

        pls = PLSRegression(n_components=actual_ncomp, scale=False)
        pls.fit(X, target)
        coef = pls.coef_.flatten()
        intercept = float(pls._y_mean - np.dot(pls._x_mean, coef))
        intercept_vector.append(intercept)

        target_col = current_idx - window_half_size

        if coef.size == (2 * k + 1):
            transfer_matrix[start_idx:end_idx, target_col] = coef[:actual_window_size]
        else:
            transfer_matrix[start_idx:end_idx, target_col] = coef[pad_left:pad_left + actual_window_size]

        processed_count += 1
        if verbose and (current_idx % 100 == 0 or processed_count == total_to_process):
            progress = processed_count / total_to_process * 100
            print(f"\rProcessing wavelength {current_idx}/{n_wavelengths}: {progress:.1f}%", end='')

        current_idx += 1

    if verbose:
        print(f"\nPDS processing completed. {processed_count}/{total_to_process} wavelengths processed.")


    left_pad = np.zeros((n_wavelengths, window_half_size - 1))
    right_pad = np.zeros((n_wavelengths, window_half_size - 1))
    full_transfer_matrix = np.hstack((left_pad, transfer_matrix, right_pad))  # shape: (n_wave, n_wave)

    intercept_arr = np.array(intercept_vector).flatten()
    left_intercept_pad = np.zeros(window_half_size - 1)
    right_intercept_pad = np.zeros(window_half_size - 1)
    full_intercept = np.concatenate((left_intercept_pad, intercept_arr, right_intercept_pad))  # shape: (n_wave,)

    # ========== slave_spectra ==========
    # slave_spectra: (samples x wavelengths)
    # full_transfer_matrix: (wavelengths x wavelengths)
    # full_intercept: (wavelengths,)
    corrected_slave = slave_spectra @ full_transfer_matrix + full_intercept

    return corrected_slave



def snv_correction(X):

    mean_vals = np.mean(X, axis=1, keepdims=True)  
    std_vals = np.std(X, axis=1, keepdims=True)    
    

    std_vals[std_vals == 0] = 1.0
    
    # SNV
    X_snv = (X - mean_vals) / std_vals
    return X_snv

def generate_segment_ids(wavelengths, num_segments=3):
    """
    Generates segment IDs based on band division (e.g., 3 segments: 0, 1, 2)
    - wavelengths: List[float] or ndarray, with a length of num_bands
    - num_segments: int, the number of segments (default: 3)

    Returns:
    segment_ids: Tensor [seq_len], the segment index for each band
    """
    wavelengths = torch.tensor(wavelengths, dtype=torch.float32)
    min_wl, max_wl = wavelengths.min(), wavelengths.max()
    segment_edges = torch.linspace(min_wl, max_wl, steps=num_segments + 1)

    segment_ids = torch.zeros_like(wavelengths, dtype=torch.long)
    for i in range(num_segments):
        in_segment = (wavelengths >= segment_edges[i]) & (wavelengths < segment_edges[i + 1])
        segment_ids[in_segment] = i
    segment_ids[wavelengths == max_wl] = num_segments - 1  
    return segment_ids  # shape [seq_len]