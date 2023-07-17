import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import dct
plt.rcParams["font.family"]="STSong"
# 执行 file2.py 文件中的代码
from 样本熵 import SampEn


framerate = 200 #信号采样率
se_if_list = []
for i in range(1,10):
    if i <=7 or i==9:
       for j in range(1,21):
           filename = "附件" + str(i) +"/" +str(j) + '.txt'
           signal = np.genfromtxt(filename,dtype = float)
           signal = signal * 1.0 / (max(abs(signal)))  # 归一化
           signal_len = len(signal)
           signal_add = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])  # 预加重
           time = np.arange(0, signal_len) / 1.0 * framerate
           wlen = 512
           inc = 128
           N = 512
           if signal_len < wlen:
               nf = 1
           else:
               nf = int(np.ceil((1.0 * signal_len - wlen + inc) / inc))
           pad_len = int((nf - 1) * inc + wlen)
           zeros = np.zeros(pad_len - signal_len)
           pad_signal = np.concatenate((signal, zeros))
           indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
           indices = np.array(indices, dtype=np.int32)
           frames = pad_signal[indices]
           win = np.hanning(wlen)
           m = 24
           s = np.zeros((nf, m))
           for c in range(nf):
               x = frames[c:c + 1]
               y = win * x[0]
               a = np.fft.fft(y)
               b = np.square(abs(a))
               mel_high = 1125 * np.log(1 + (framerate / 2) / 700)
               mel_point = np.linspace(0, mel_high, m + 2)
               Fp = 700 * (np.exp(mel_point / 1125) - 1)
               w = int(N / 2 + 1)
               df = framerate / N
               fr = []
               for o in range(w):
                   frs = int(o * df)
                   fr.append(frs)
               melbank = np.zeros((m, w))
               for p in range(m + 1):
                   f1 = Fp[p - 1]
                   f2 = Fp[p + 1]
                   f0 = Fp[p]
                   n1 = np.floor(f1 / df)
                   n2 = np.floor(f2 / df)
                   n0 = np.floor(f0 / df)
                   for l in range(w):
                       if l >= n1 and l <= n0:
                           melbank[p - 1, l] = (l - n1) / (n0 - n1)
                       if l >= n0 and l <= n2:
                           melbank[p - 1, l] = (n2 - l) / (n2 - n0)
                   for v in range(w):
                       s[c, p - 1] = s[c, p - 1] + b[v:v + 1] * melbank[p - 1, v]
           logs = np.log(s)
           num_ceps = 12
           D = dct(logs, type=2, axis=0, norm='ortho')[:, 1: (num_ceps + 1)]
           D = pd.DataFrame(D)
           a = D.shape[0]
           b = len(D.columns)
           dtm = np.zeros((a, b))
           dtm = pd.DataFrame(dtm)

           for d in range(2, (a - 3)):
               dtm.iloc[d,] = -2 * D.iloc[d - 2,] - D.iloc[d - 1,] + D.iloc[d + 1,] + 2 * D.iloc[d + 2,]
           dtm = dtm / 3

           dtm_m = np.zeros((a, b))
           dtm_m = pd.DataFrame(dtm_m)
           for e in range(2, (a - 3)):
               dtm_m.iloc[e,] = -2 * dtm.loc[e - 2,] - dtm.iloc[e - 1,] + dtm.iloc[e + 1,] + 2 * dtm.iloc[e + 2,]
           dtm_m = dtm_m / 3
           result = pd.concat([D, dtm, dtm_m], axis=1)
           result = result.iloc[2:-3]
           # 获取第1列、第9列和第25列的特征值
           first_col = result.iloc[:, 0].values
           ninth_col = result.iloc[:, 8].values
           twentyfifth_col = result.iloc[:, 24].values
           # 置于新的数据框中
           new_df = pd.DataFrame({
               'First Column': first_col,
               'Ninth Column': ninth_col,
               'Twenty-Fifth Column': twentyfifth_col
           })

           for k in range(3):
              se_if = SampEn(new_df.iloc[:,k].values,2,0.2*np.std(new_df.iloc[:,k].values))
              se_if_list.append(se_if)
    else:
        for j in range(1, 31):
            filename = "附件" + str(i) + "/" + str(j) + '.txt'
            signal = np.genfromtxt(filename, dtype=float)
            signal = signal * 1.0 / (max(abs(signal)))  # 归一化
            signal_len = len(signal)
            signal_add = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])  # 预加重
            time = np.arange(0, signal_len) / 1.0 * framerate
            wlen = 512
            inc = 128
            N = 512
            if signal_len < wlen:
                nf = 1
            else:
                nf = int(np.ceil((1.0 * signal_len - wlen + inc) / inc))
            pad_len = int((nf - 1) * inc + wlen)
            zeros = np.zeros(pad_len - signal_len)
            pad_signal = np.concatenate((signal, zeros))
            indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (wlen, 1)).T
            indices = np.array(indices, dtype=np.int32)
            frames = pad_signal[indices]
            win = np.hanning(wlen)
            m = 24
            s = np.zeros((nf, m))
            for c in range(nf):
                x = frames[c:c + 1]
                y = win * x[0]
                a = np.fft.fft(y)
                b = np.square(abs(a))
                mel_high = 1125 * np.log(1 + (framerate / 2) / 700)
                mel_point = np.linspace(0, mel_high, m + 2)
                Fp = 700 * (np.exp(mel_point / 1125) - 1)
                w = int(N / 2 + 1)
                df = framerate / N
                fr = []
                for o in range(w):
                    frs = int(o * df)
                    fr.append(frs)
                melbank = np.zeros((m, w))
                for p in range(m + 1):
                    f1 = Fp[p - 1]
                    f2 = Fp[p + 1]
                    f0 = Fp[p]
                    n1 = np.floor(f1 / df)
                    n2 = np.floor(f2 / df)
                    n0 = np.floor(f0 / df)
                    for l in range(w):
                        if l >= n1 and l <= n0:
                            melbank[p - 1, l] = (l - n1) / (n0 - n1)
                        if l >= n0 and l <= n2:
                            melbank[p - 1, l] = (n2 - l) / (n2 - n0)
                    for v in range(w):
                        s[c, p - 1] = s[c, p - 1] + b[v:v + 1] * melbank[p - 1, v]

            logs = np.log(s)
            num_ceps = 12
            D = dct(logs, type=2, axis=0, norm='ortho')[:, 1: (num_ceps + 1)]
            D = pd.DataFrame(D)
            a = D.shape[0]
            b = len(D.columns)
            dtm = np.zeros((a, b))
            dtm = pd.DataFrame(dtm)
            for g in range(2, (a - 3)):
                dtm.iloc[g,] = -2 * D.iloc[g - 2,] - D.iloc[g - 1,] + D.iloc[g + 1,] + 2 * D.iloc[g + 2,]
            dtm = dtm / 3

            dtm_m = np.zeros((a, b))
            dtm_m = pd.DataFrame(dtm_m)
            for h in range(2, (a - 3)):
                dtm_m.iloc[h,] = -2 * dtm.loc[h - 2,] - dtm.iloc[h - 1,] + dtm.iloc[h + 1,] + 2 * dtm.iloc[h + 2,]
            dtm_m = dtm_m / 3
            result = pd.concat([D, dtm, dtm_m], axis=1)
            result = result.iloc[2:-3]
            # 获取第一列、第九列和第25列的值
            first_col = result.iloc[:, 0].values
            ninth_col = result.iloc[:, 8].values
            twentyfifth_col = result.iloc[:, 24].values
            #置于新的数据框中
            new_df = pd.DataFrame({
                'First Column': first_col,
                'Ninth Column': ninth_col,
                'Twenty-Fifth Column': twentyfifth_col
            })

            # 打印新的数据框
            print(new_df)
            for m in range(3):
                new_df.iloc[:,m] = pd.DataFrame(new_df.iloc[:,m])
                se_if = SampEn(new_df.iloc[:,m].values,2,0.2*np.std(new_df.iloc[:,m].values))
                se_if_list.append(se_if)

arr = np.array(se_if_list)
np.savetxt('新数据.txt', arr)
sam = np.loadtxt('新数据.txt')
sam = np.array(sam)
sam = sam.reshape(190,3)
sam = pd.DataFrame(sam)
sam[np.isinf(sam)] = np.nan
sam = sam.fillna(method= 'bfill')
np.savetxt('MFCC样本熵.txt',sam)


