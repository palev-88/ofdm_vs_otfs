"""
QAM Modulation / Demodulation utilities.

Supports QPSK (4QAM), 16QAM, 64QAM.
Gray-coded constellations, unit average power.
Hard-decision demapping for uncoded BER.
"""

import numpy as np
from typing import Tuple

# ═══════════════════════════════════════════════════════════════
# Constellation generation (Gray-coded, unit power)
# ═══════════════════════════════════════════════════════════════

def qam_constellation(order: int) -> Tuple[np.ndarray, int]:
    """
    Generate Gray-coded QAM constellation with unit average power.
    
    Args:
        order: Modulation order (4=QPSK, 16=16QAM, 64=64QAM)
    
    Returns:
        constellation: Complex constellation points indexed by Gray code
        bits_per_symbol: log2(order)
    """
    if order == 4:  # QPSK
        bps = 2
        c = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
    elif order == 16:
        bps = 4
        levels = np.array([-3, -1, 3, 1])  # Gray: 00→-3, 01→-1, 10→+3, 11→+1
        c = np.zeros(16, dtype=complex)
        for i in range(4):
            for q in range(4):
                idx = i * 4 + q
                c[idx] = complex(levels[i], levels[q])
        c /= np.sqrt(np.mean(np.abs(c)**2))
    elif order == 64:
        bps = 6
        levels = np.array([-7, -5, -1, -3, 7, 5, 1, 3])  # Gray coded
        c = np.zeros(64, dtype=complex)
        for i in range(8):
            for q in range(8):
                idx = i * 8 + q
                c[idx] = complex(levels[i], levels[q])
        c /= np.sqrt(np.mean(np.abs(c)**2))
    else:
        raise ValueError(f"Unsupported QAM order: {order}")
    
    return c, bps


def qam_modulate(bits: np.ndarray, order: int) -> np.ndarray:
    """Map bits to QAM symbols."""
    constellation, bps = qam_constellation(order)
    n_symbols = len(bits) // bps
    bits = bits[:n_symbols * bps]
    
    # Pack bits into symbol indices
    indices = np.zeros(n_symbols, dtype=int)
    for b in range(bps):
        indices += bits[b::bps].astype(int) << (bps - 1 - b)
    
    return constellation[indices]


def qam_demodulate(symbols: np.ndarray, order: int) -> np.ndarray:
    """Hard-decision QAM demapping (minimum distance)."""
    constellation, bps = qam_constellation(order)
    n_symbols = len(symbols)
    
    # Find closest constellation point
    distances = np.abs(symbols[:, None] - constellation[None, :])  # (N, M)
    indices = np.argmin(distances, axis=1)
    
    # Unpack indices to bits
    bits = np.zeros(n_symbols * bps, dtype=int)
    for b in range(bps):
        bits[b::bps] = (indices >> (bps - 1 - b)) & 1
    
    return bits


def generate_random_bits(n_bits: int, rng=None) -> np.ndarray:
    """Generate random bits."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=n_bits)


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("═══ QAM Smoke Test ═══\n")
    
    for order in [4, 16, 64]:
        c, bps = qam_constellation(order)
        avg_power = np.mean(np.abs(c)**2)
        
        # Round-trip test
        bits = generate_random_bits(bps * 1000)
        symbols = qam_modulate(bits, order)
        bits_rx = qam_demodulate(symbols, order)
        ber = np.mean(bits != bits_rx)
        
        print(f"  {order}QAM: {bps} bps, avg_power={avg_power:.4f}, "
              f"round-trip BER={ber:.0e} ({'✓' if ber == 0 else '✗'})")
    
    # Test with noise
    bits = generate_random_bits(2 * 10000)
    symbols = qam_modulate(bits, 4)
    noise = 0.3 * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
    bits_rx = qam_demodulate(symbols + noise, 4)
    ber = np.mean(bits != bits_rx)
    print(f"\n  QPSK with noise: BER={ber:.4f} (should be small but nonzero)")
