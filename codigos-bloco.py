import numpy as np
import komm
import matplotlib.pyplot as plt

cod = komm.BlockCode(parity_submatrix=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])

(n, k, d) = (cod.length, cod.dimension, cod.minimum_distance)
print(n, k, d)

G = cod.generator_matrix
H = cod.parity_check_matrix

cod.codeword_table

cod.codeword_weight_distribution  # A_w

LUT = cod.coset_leader_table

cod.coset_leader_weight_distribution  # α_w

b = np.array([1, 0, 0, 1, 1, 0])
# s = (b @ H.T) % 2
# idx = komm.binlist2int(s)
# e_hat = LUT[idx, :]
# v_hat = (b + e_hat) % 2
# u_hat = v_hat[:k]
# u_hat
u_hat = cod.decode(b, 'syndrome_table')
print(u_hat)

################################
# Curca de BER (bit error rate)#
################################

Ncw = 10000
code = komm.HammingCode(3)
mod = komm.PSKModulation(2)
awgn = komm.AWGNChannel()
EbNo_dB = np.arange(-1.0, 8.0)
EbNo = 10.0**(EbNo_dB / 10.0)

(n, k, R) = (code.length, code.dimension, code.rate)
u = np.random.randint(0, 2, size=(Ncw, k))
v = np.apply_along_axis(code.encode, axis=1, arr=u)
v_seq = v.flat[:]
x = mod.modulate(v_seq)

decode_hard = lambda x: code.decode(x, method='syndrome_table')
decode_soft = lambda x: code.decode(x, method='exhaustive_search_soft')

BER_hard = []
BER_soft = []
for gamma in EbNo:
    awgn.snr = gamma * R  # Modulação complexa.
    y = awgn(x)

    # HDD
    b_seq = mod.demodulate(y)
    b = np.reshape(b_seq, newshape=(Ncw, -1))
    u_hat = np.apply_along_axis(decode_hard, axis=1, arr=b)
    BER_hard.append(np.mean(np.bitwise_xor(u, u_hat)))

    # SDD
    sb_seq = y  #mod.demodulate(y, decision_method='soft')
    sb = np.reshape(sb_seq, newshape=(Ncw, -1))
    u_hat = np.apply_along_axis(decode_soft, axis=1, arr=sb)
    BER_soft.append(np.mean(np.bitwise_xor(u, u_hat)))


plt.semilogy(EbNo_dB, BER_hard)
plt.semilogy(EbNo_dB, BER_soft)
plt.semilogy(EbNo_dB, komm.qfunc(np.sqrt(2*EbNo)), 'k--')
plt.grid()
plt.ylim(3e-5, 0.2)
plt.show()