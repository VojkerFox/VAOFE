import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np
import os

def generate_setup_chart(symbol, rates, target_level, pivot_price, direction, output_path="setup_snapshot.png"):
    """
    Piirtää graafin ja tallentaa sen kuvana.
    rates: list of dicts (MT5 rates)
    target_level: BOS-taso (oranssi)
    pivot_price: Pivot-kimmokkeen hinta (keltainen/punainen merkintä)
    """
    # 1. Muutetaan data DataFrameksi
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # 2. Määritetään visuaalinen tyyli
    style = mpf.make_mpf_style(base_mpf_style='nightclouds', gridstyle='', y_on_right=True)
    
    # 3. Luodaan tasot piirrettäväksi
    hlines = dict(hlines=[target_level], colors=['orange'], linestyle='-.', linewidths=1.5)
    
    # 4. Piirretään graafi
    fig, axlist = mpf.plot(df, type='candle', style=style, 
                           title=f"{symbol} - VAOFE Sniper Setup",
                           hlines=hlines,
                           returnfig=True, figsize=(10, 6))
    
    # Lisätään Pivot-merkintä (Entry-taso)
    axlist[0].axhline(y=pivot_price, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
    
    # Tallenna
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path